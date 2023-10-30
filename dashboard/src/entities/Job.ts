export interface Job {
    createdAt: string;
    endedAt: string;
    id: string;
    startedAt: string;
    state: string;
    updatedAt: string;
    name: string;
    experimentId?: string;
}

export interface DatasetPayload {
    role: string,
    datasetGroups: {
        [key:string]: string[]
    }
}

export interface DatasetControls {
    label: string,
    controls: string[]
}

