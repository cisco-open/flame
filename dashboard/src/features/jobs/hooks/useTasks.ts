import { useQuery } from "@tanstack/react-query";
import { LOGGEDIN_USER } from "../../../constants";
import { Task } from "../../../entities/Task";
import ApiClient from "../../../services/api-client";

const useTasks = (id: string) => {
    const apiClient = new ApiClient<Task[]>(`users/${LOGGEDIN_USER.name}/jobs/${id}/tasks`);
    return useQuery({
        enabled: !!id,
        queryKey: ['tasks', id],
        queryFn: apiClient.getAll,
    });
}

export default useTasks;