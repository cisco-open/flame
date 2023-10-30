export interface ExperimentData {
    experiments?: Experiment[];
    experiment?: Experiment;
}

export interface Experiment {
    artifact_location: string;
    experiment_id: string;
    lifecycle_state: string;
    name: string;
    last_update_time: number;
    creation_time: number;
}