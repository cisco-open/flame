export interface JobForm {
    design: string | undefined;
    hyperParameters: any | undefined;
    basemodelName: string | undefined;
    basemodelVersion: string | undefined;
    backend: string | undefined;
    maxRunTime: string | undefined;
    priority: string | undefined;
    datasets: string | undefined;
    dependencies: string | undefined;
    optimizerName: string | undefined;
    optimizerKwargs: string | undefined;
    selectorKwargs: string | undefined;
    selectorName: string | undefined;
    designId?: string | undefined;
    dataSpec: any;
}