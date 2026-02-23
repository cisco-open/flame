import os
import sys
import time
import logging
import argparse
import torch
import numpy as np
import gc
from datetime import datetime
from torch.nn import CrossEntropyLoss
import sklearn.metrics
from sklearn.metrics import matthews_corrcoef, confusion_matrix

# Add workspace roots to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "lib/python")))

from examples.fwdllm.data_preprocessing.text_classification_preprocessor import TLMPreprocessor
from examples.fwdllm.trainer.model.transformer.model_args import ClassificationArgs
from examples.fwdllm.data_manager.text_classification_data_manager import TextClassificationDataManager
from examples.fwdllm.data_manager.base_data_manager import BaseDataManager
from examples.fwdllm.expts.initializer import set_seed, create_model
from flame.config import Config
import contextlib

logger = logging.getLogger(__name__)

# Reconstructing the timer decorator as requested
def timer_decorator(func):
    def wrapper(*args, **kwargs):
        # Adding cuda synchronize before timing if cuda is used
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        
        duration = end_time - start_time
        # Format consistent with historical logs: [GJD] In timer_decorator wrapper
        logger.info(f"[GJD] In timer_decorator wrapper - {func.__name__} took {duration:.4f}s")
        return result
    return wrapper

class MockAggregator:
    def __init__(self, model, test_data_global, num_labels, device, args):
        self.model = model
        self.test_global = test_data_global
        self.num_labels = num_labels
        self.device = device
        self.args = args # Should contain eval_batch_size
    
    def log_memory(self, msg, device):
        if device.type == 'cuda':
            logger.debug(f"{msg} - CUDA Memory Allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")

    @timer_decorator
    def eval_model(self, epoch=0, global_step=0, device=None):
        if not device:
            device = self.device

        logger.info(f"device inside eval_model() is set to: {device}")
        self.log_memory("start eval_model", device)

        results = {}
        eval_loss_acc = torch.tensor(0.0, device=device)
        nb_eval_steps = 0
        test_sample_len = len(self.test_global.dataset)
        
        # Move model to device before performing the eval
        self.model.to(device)
        self.model.eval()

        # One-time GPU data transfer if enabled and not already cached
        t_cache_start = time.time()
        if not hasattr(self, "_cached_test_data") or self._cached_test_data is None:
            logger.info("One-time GPU data transfer for evaluation dataset")
            self._cached_test_data = [t.to(device) for t in self.test_global.dataset.tensors]
        
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_cache_end = time.time()
        cache_time = t_cache_end - t_cache_start

        # Use GPU tensors for inputs/mask/labels
        input_ids_all = self._cached_test_data[1]
        input_mask_all = self._cached_test_data[2]
        labels_all = self._cached_test_data[4]

        logger.info(f"Actual tensor shape in GPU memory: {input_ids_all.shape} (BS, SeqLen)")

        # Accumulate predictions on GPU
        preds_gpu = torch.empty((test_sample_len, self.num_labels), device=device)
        out_label_ids_gpu = torch.empty(test_sample_len, dtype=labels_all.dtype, device=device)

        batch_size = self.args.eval_batch_size
        loss_fct = CrossEntropyLoss()

        # Detailed timing
        inner_loop_times = []

        t_loop_start = time.time()
        from torch.cuda.amp import autocast
        autocast_cm = autocast() if self.args.fp16 else contextlib.nullcontext()
        with torch.no_grad(), autocast_cm:
            for i in range(0, test_sample_len, batch_size):
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t_inner_start = time.time()
                
                end_index = min(i + batch_size, test_sample_len)
                
                x = input_ids_all[i:end_index]
                mask = input_mask_all[i:end_index]
                labels = labels_all[i:end_index]

                # Forward pass - PASSING MASK NOW
                output = self.model(x, attention_mask=mask)
                
                if hasattr(output, "logits"):
                    logits = output.logits
                elif isinstance(output, (tuple, list)):
                    logits = output[0]
                else:
                    logits = output

                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                eval_loss_acc += loss

                preds_gpu[i:end_index] = logits
                out_label_ids_gpu[i:end_index] = labels
                nb_eval_steps += 1
                
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t_inner_end = time.time()
                inner_loop_times.append(t_inner_end - t_inner_start)
        
        t_loop_end = time.time()

        # Post-processing timing
        t_post_start = time.time()
        eval_loss = (eval_loss_acc / nb_eval_steps).item()
        preds = preds_gpu.cpu().numpy()
        out_label_ids = out_label_ids_gpu.cpu().numpy()

        model_outputs = preds
        preds_argmax = np.argmax(preds, axis=1)
        
        result, wrong = self.compute_metrics(
            preds_argmax, out_label_ids, self.test_global.examples
        )
        result["eval_loss"] = eval_loss
        results.update(result)
        t_post_end = time.time()

        # Cleanup timing
        t_clean_start = time.time()
        torch.cuda.empty_cache()
        gc.collect()
        t_clean_end = time.time()

        # Final breakdown
        logger.info(f"Breakdown: Cache: {cache_time*1000:.2f}ms, Total Loop: {(t_loop_end-t_loop_start)*1000:.2f}ms, Avg Batch: {np.mean(inner_loop_times)*1000:.2f}ms, Post-proc: {(t_post_end-t_post_start)*1000:.2f}ms, Cleanup: {(t_clean_end-t_clean_start)*1000:.2f}ms")

        return result, model_outputs, wrong

    def compute_metrics(self, preds, labels, eval_examples=None):
        assert len(preds) == len(labels)
        self.log_memory("start compute_metrics", self.device)

        extra_metrics = {}
        extra_metrics["acc"] = sklearn.metrics.accuracy_score(labels, preds)
        # mismatched = labels != preds

        # if eval_examples:
        #     wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]
        # else:
        #     wrong = ["NA"]
        wrong = [] # Simplified for benchmark

        mcc = matthews_corrcoef(labels, preds)

        # confusion_matrix logic
        try:
             tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        except ValueError:
             tn, fp, fn, tp = 0, 0, 0, 0

        self.log_memory("end compute_metrics", self.device)

        return (
            {**{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}, **extra_metrics},
            wrong,
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    parser.add_argument("--batch_sizes", type=str, default="8,256", help="Comma separated batch sizes to test")
    parser.add_argument("--test_cut_off", type=int, default=7600, help="Number of samples from test set")
    args = parser.parse_args()

    # Log setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d - %(funcName)s(): %(message)s",
        datefmt="%Y-%m-%d,%H:%M:%S",
    )

    config = Config(args.config)
    set_seed(config.hyperparameters.manual_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load attributes to get num_labels
    attributes = BaseDataManager.load_attributes(config.hyperparameters.data_file_path)
    num_labels = len(attributes["label_vocab"])

    # Create original model and tokenizer
    model_args = ClassificationArgs()
    model_args.model_name = config.hyperparameters.model_name
    model_args.model_type = config.hyperparameters.model_type
    model_args.load(model_args.model_name)
    model_args.num_labels = num_labels
    model_args.update_from_dict(vars(config.hyperparameters))
    model_args.config["num_labels"] = num_labels
    
    # Ensure eval_batch_size is set (will be overridden in loop)
    model_args.eval_batch_size = 8

    model_config, client_model, tokenizer = create_model(
        model_args, formulation="classification"
    )

    # Data management
    preprocessor = TLMPreprocessor(
        args=model_args, label_vocab=attributes["label_vocab"], tokenizer=tokenizer
    )
    
    batch_sizes = [int(bs.strip()) for bs in args.batch_sizes.split(",")]
    
    for bs in batch_sizes:
        logger.info(f"\n{'='*20} Testing Batch Size: {bs} {'='*20}")
        model_args.eval_batch_size = bs
        
        dm = TextClassificationDataManager(
            config.hyperparameters,
            model_args,
            preprocessor,
            0,
            1
        )
        # Use load_federated_data(process_id=0) for server-side global test data
        (
            _, _, test_data_global,
            _, _, _, _
        ) = dm.load_federated_data(process_id=0, test_cut_off=args.test_cut_off)

        logger.info(f"Dataset size: {len(test_data_global.dataset)}")
        
        agg = MockAggregator(client_model, test_data_global, num_labels, device, model_args)
        
        # Warmup
        if device.type == 'cuda':
            logger.info("Warming up...")
            agg.eval_model() 
        
        logger.info(f"Running timed eval for BS={bs}...")
        agg.eval_model()

if __name__ == "__main__":
    main()
