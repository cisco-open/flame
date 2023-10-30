import { createContext } from "react";
import { Design } from "../../entities/Design";

export interface DesignContextType {
    designInEdit: Design;
}

export const DesignContext = createContext<DesignContextType>({} as DesignContextType);