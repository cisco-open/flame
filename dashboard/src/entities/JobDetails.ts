export interface GetRunsPayload {
    experiment_ids: string[];
    max_results: number;
    order_by: string[];
    run_view_type: RunViewType;
}

export enum RunViewType {
    activeOnly = 'ACTIVE_ONLY',
    deletedOnly = 'DELETED_ONLY',
}

export interface RunResponse {
    runs: Run[];
}

export interface Run {
    data: RunData;
    info: RunInfo;
    startDate: string;
    endDate: string;
    taskId: string;
}

export interface RunData {
    metrics: Metric[];
    parameters: Parameter[];
    tags: Tag[];
}

export interface RunInfo {
    artifact_uri: string;
    end_time: number;
    experiment_id: string;
    lifecycle_stage: string;
    run_id: string;
    run_name: string;
    run_uuid: string;
    start_time: number;
    status: string;
    user_id: string;
}

export interface Metric {
    key: string;
    value: number;
    timestamp: number;
    step: number;
}

export interface Parameter {
    key: string;
    value: string;
}

export interface Tag {
    key: string;
    value: string;
}