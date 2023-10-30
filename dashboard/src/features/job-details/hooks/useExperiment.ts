import { useQuery } from "@tanstack/react-query";
import { ExperimentData } from "../../../entities/Experiment";
import ApiClient from "../../../services/api-client";

const useExperiment = (jobId: string) => {
  const apiClient = new ApiClient<ExperimentData>(`experiments/get-by-name?experiment_name=${jobId}`, true);
  return useQuery({
    enabled: !!jobId,
    queryKey: ['experiment', jobId],
    queryFn: apiClient.getAll,
  });
}

export default useExperiment;