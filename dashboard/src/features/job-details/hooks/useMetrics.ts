import { useQuery } from "@tanstack/react-query";
import { ExperimentData } from "../../../entities/Experiment";
import ApiClient from "../../../services/api-client";
import { MetricsRequestParams } from "../types";

const useMetrics = (data: MetricsRequestParams) => {
  const apiClient = new ApiClient<ExperimentData>('/metrics/get-history', true);
  return useQuery({
    enabled: !!(data.metric_key && data.run_uuid),
    queryKey: ['metrics', data.metric_key],
    queryFn: () => apiClient.getAll({ params: data }),
  });
}

export default useMetrics;