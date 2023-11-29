/**
 * Copyright 2023 Cisco Systems, Inc. and its affiliates
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalCloseButton,
  ModalBody,
  Button,
  Box,
  Stepper,
  Step,
  StepIcon,
  StepIndicator,
  StepNumber,
  StepSeparator,
  StepStatus,
  StepTitle,
  useSteps
} from "@chakra-ui/react"
import { yupResolver } from "@hookform/resolvers/yup";
import React, { useEffect, useRef, useState } from "react";
import { useForm } from "react-hook-form";
import useDatasets from "../../../../hooks/useDatasets";
import useDesigns from "../../../design/hooks/useDesigns";
import { BACKEND_OPTIONS, DEFAULT_HYPERPARAMETERS } from "../../../jobs/constants";

import './JobFormModal.css';
import * as yup from 'yup';
import useDesign from "../../../design-details/hooks/useDesign";
import GeneralForm from "../general-form/GeneralForm";
import DatasetForm from "../dataset-form/DatasetForm";
import ModelSpecForm from "../model-spec-form/ModelSpecForm";
import useJobs from "../../hooks/useJobs";
import { createSaveJobPayload, getHyperparametersFromJob, mapJobToForm } from "../../utils";
import { DatasetControls, DatasetPayload } from "../../../../entities/Job";
import { JobContext } from "../../JobContext";
import { KwargPayload } from "../optimizer-form/OptimizerForm";
import { JobForm } from "../../types";

interface Props {
  isOpen: boolean;
  job: any;
  onClose: () => void;
}

interface Hyperparameter {
  key: string,
  value: string,
  id: number,
}

export interface FormFields {
  name: string,
  design: string | undefined,
  hyperparameters: string | undefined,
  basemodelName: string | undefined,
  basemodelVersion: string | undefined,
  backend: string | undefined,
  maxRunTime: string | undefined,
  priority: string | undefined,
  datasets: string | undefined,
  optimizerName: string | undefined,
  optimizerKwargs: string | undefined,
  selectorKwargs: string | undefined,
  selectorName: string | undefined,
}

const steps = ['GENERAL', 'DATASETS', 'MODEL'];

const JobFormModal = ({ isOpen, job, onClose }: Props) => {
  const initialRef: React.MutableRefObject<null> = useRef(null);
  const [ selectedDesignId, setSelectedDesignId ] = useState('');
  const [ datasetControls, setDatasetControls ] = useState<DatasetControls[] | undefined>([]);
  const [ hyperParameters, setHyperParameters ] = useState<Hyperparameter[]>(DEFAULT_HYPERPARAMETERS as Hyperparameter[]);
  const [ mappedHyperParameters, setMappedHyperParameters ] = useState<any>();

  const [ selectorKwargsPayload, setSelectorKwargsPayload ] = useState<KwargPayload | undefined>();
  const [ optimizerKwargsPayload, setOptimizerKwargsPayload ] = useState<KwargPayload | undefined>();
  const [ mappedJobToForm, setMappedJobToForm ] = useState<JobForm | null>(null);

  const [ datasetPayload, setDatasetPayload ] = useState<DatasetPayload | null>(null);
  const { data: design } = useDesign(selectedDesignId)
  const { data: designs } = useDesigns();
  const { data: datasets } = useDatasets();
  const { createMutation, editMutation } = useJobs(job?.id, onClose);
  const { activeStep, setActiveStep } = useSteps({
    index: 0,
    count: steps.length,
  });
  const backendOptions = BACKEND_OPTIONS;
  const defaultBackendOption = {
    name: 'mqtt',
    id: 1
  };

  useEffect(() => {
    if (!job) { return; }
    setMappedJobToForm(mapJobToForm(job));
    setHyperParameters(getHyperparametersFromJob(job) as unknown as any);
  }, [job])

  useEffect(() => {
    const mappedHyperparameters = hyperParameters.reduce((acc: any, param: any) => {
      acc[param.key] = param.value;
      return acc;
    }, {});
    setMappedHyperParameters(mappedHyperparameters);
  }, [hyperParameters])

  useEffect(() => {
    if (!design) { return; }
    const roles = design?.schema?.roles.filter(role => role.isDataConsumer);
    const dictionary = {} as unknown as any;
    const datasetControls = roles?.map(role => {
      role.groupAssociation.map(group => {
        for (let i = 0; i < Object.keys(group).length; i++) {
          const key = Object.keys(group)[i];
          if (!dictionary.hasOwnProperty(key)) {
            dictionary[key] = [group[key]]
          } else {
            dictionary[key] = [...dictionary[key], group[key]]
          }
        }
      });

      const lengths = Object.keys(dictionary).map(key => ({
        label: role.name,
        controls: Array.from(new Set(dictionary[key])) as string[]
      }));

      const index: number = lengths
        .map(list => list.controls.length)
        .indexOf(Math.max(...lengths.map(list => list.controls.length)));

      return lengths[index];
    });

    setDatasetControls(datasetControls);
  }, [design]);

  const handleClose = () => {
    onClose();
    reset();
    setSelectedDesignId('');
    setHyperParameters(DEFAULT_HYPERPARAMETERS as Hyperparameter[]);
    setActiveStep(0);
  }

  const onSave = () => {
    createMutation.mutate(createSaveJobPayload({
      ...getValues(),
      designId: selectedDesignId,
      hyperParameters,
      dataSpec: datasetPayload,
      modelSpec: {
        selector: selectorKwargsPayload,
        optimizer: optimizerKwargsPayload,
        hyperParameters: mappedHyperParameters,
      }
    }));
    handleClose();
  }

  const onEdit = () => {
    editMutation.mutate(createSaveJobPayload({
      ...getValues(),
      designId: selectedDesignId,
      hyperParameters,
      dataSpec: datasetPayload,
      modelSpec: {
        selector: selectorKwargsPayload,
        optimizer: optimizerKwargsPayload,
        hyperParameters: mappedHyperParameters,
      }
    }));
  }

  const schema = yup.object().shape({
    name: yup.string().required(),
    design: yup.string(),
    hyperparameters: yup.string(),
    basemodelName: yup.string(),
    basemodelVersion: yup.string(),
    backend: yup.string(),
    maxRunTime: yup.string(),
    priority: yup.string(),
    datasets: yup.string(),
    optimizerName: yup.string(),
    optimizerKwargs: yup.string(),
    selectorKwargs: yup.string(),
    selectorName: yup.string(),
  });

  const { register, handleSubmit, formState, reset, getValues, setValue } = useForm({
    resolver: yupResolver(schema),
    defaultValues: {
      backend: defaultBackendOption.name,
      maxRunTime: '3600',
    }
  });

  const handleNext = () => {
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
  }

  const handlePrevious = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  }

  return (
    <Modal
      initialFocusRef={initialRef}
      isOpen={isOpen}
      onClose={handleClose}
      size="full"
    >
      <ModalOverlay />
      <ModalContent height="100%">
        <ModalHeader textAlign="center">{job ? 'EDIT JOB' : 'CREATE JOB'}</ModalHeader>

        <ModalCloseButton />

        <JobContext.Provider value={{ setSelectorKwargsPayload, setOptimizerKwargsPayload, job }}>
          <ModalBody display="flex" flexDirection="column" gap="10px" height="calc(100% - 62px)">
            <Box>
              <Stepper index={activeStep}>
                {steps.map((step, index) => (
                  <Step key={index}>
                    <StepIndicator>
                      <StepStatus
                        complete={<StepIcon />}
                        incomplete={<StepNumber />}
                        active={<StepNumber />}
                      />
                    </StepIndicator>

                    <Box flexShrink='0'>
                      <StepTitle>{step}</StepTitle>
                    </Box>

                    <StepSeparator />
                  </Step>
                ))}
              </Stepper>
            </Box>

            {
              activeStep === 0 &&
              <GeneralForm
                designs={designs}
                backendOptions={backendOptions}
                setSelectedDesignId={setSelectedDesignId}
                selectedDesignId={selectedDesignId}
                register={register}
                setValue={(name, value) => setValue(name as unknown as any, value)}
              />
            }

            {
              activeStep === 1 &&
              <DatasetForm
                datasetControls={datasetControls}
                datasets={datasets}
                setDatasetPayload={setDatasetPayload}
                mappedJobToForm={mappedJobToForm}
              />
            }

            {
              activeStep === 2 &&
              <ModelSpecForm
                register={register}
                hyperParameters={hyperParameters}
                setHyperParameters={(params) => setHyperParameters([...params])}
                setValue={(name, value) => setValue(name as unknown as any, value)}
              />
            }

            <Box display="flex" justifyContent="space-between">
              <Button isDisabled={activeStep === 0} colorScheme='blue' onClick={handlePrevious}>Previous</Button>
              {
                activeStep !== steps.length - 1 &&
                <Button
                  isDisabled={(activeStep === 0 && !selectedDesignId) || activeStep === 1 && !datasetPayload}
                  colorScheme='blue'
                  onClick={handleNext}
                >
                  Next
                </Button>
              }
              { activeStep === steps.length - 1 &&
                <Button onClick={job ? onEdit : onSave} colorScheme='blue' mr={3}>
                  Save
                </Button>
              }
            </Box>
          </ModalBody>
        </JobContext.Provider>
      </ModalContent>
    </Modal>
  )
}

export default JobFormModal;