import { useMutation } from "@tanstack/react-query";
import { RunResponse } from "../../../entities/JobDetails";
import ApiClient from "../../../services/api-client";

const useRuns = () => {
  const apiClient = new ApiClient<RunResponse>('runs/search', true);
  return useMutation({
    mutationKey: ['runs'],
    mutationFn: apiClient.post,
  });
}

export default useRuns;