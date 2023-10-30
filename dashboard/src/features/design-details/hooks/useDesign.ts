import { useQuery } from "@tanstack/react-query";
import { LOGGEDIN_USER } from "../../../constants";
import { DesignDetails } from "../../../entities/DesignDetails";
import ApiClient from "../../../services/api-client";

const apiClient = new ApiClient<DesignDetails>(`users/${LOGGEDIN_USER.name}/designs`);

const useDesign = (id: string) => useQuery({
    enabled: !!id,
    queryKey: ['design', id],
    queryFn: () => apiClient.get(id),
});

export default useDesign;