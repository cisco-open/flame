import { createContext } from "react";
import { KwargPayload } from "./components/optimizer-form/OptimizerForm";

export interface JobContextType {
    setSelectorKwargsPayload: (data: KwargPayload) => void;
    setOptimizerKwargsPayload: (data: KwargPayload) => void;
    job: any;
}

export const JobContext = createContext<JobContextType>({} as JobContextType);