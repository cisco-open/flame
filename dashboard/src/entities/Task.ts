export interface Task {
    computeId: string,
    jobId: string,
    key: string,
    log: string
    role: string,
    state: string,
    taskId: string,
    timestamp: string,
    type: string,
    level: number,
    groupAssociation: any,
    count?: number,
    group?: string[],
}