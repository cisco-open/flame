import { useQuery } from "@tanstack/react-query";
import { LOGGEDIN_USER } from "../constants";
import { Dataset } from "../entities/Dataset";
import ApiClient from "../services/api-client";


const useCodes = (designId: string) => {
    const apiClient = new ApiClient<File>(`users/${LOGGEDIN_USER.name}/designs/${designId}/code`);
    const xmlHttp = new XMLHttpRequest();
    xmlHttp.open('GET', `api/users/${LOGGEDIN_USER.name}/designs/${designId}/code`)
    xmlHttp.overrideMimeType('text/plain; charset=x-user-defined');

    return xmlHttp.send();

    // return useQuery({
    //     queryKey: ['datasets'],
    //     queryFn: () => apiClient.getAll({ responseType: 'arraybuffer', overrideMimeType: "text/plain; charset=x-user-defined" }),
    // })
}

export default useCodes;