import { LOGGEDIN_USER } from '../../constants';
import { Dataset } from '../../entities/Dataset';
import { Experiment } from '../../entities/Experiment';
import { DatasetPayload, Job } from '../../entities/Job';
import { KwargPayload, OptimizerGroup } from './components/optimizer-form/OptimizerForm';

export const mapJobsWithExperimentId = (jobs: Job[], experiments: Experiment[] | undefined) => jobs?.map(job => ({
    ...job,
    experimentId: experiments?.find(experiment => experiment.name === job.id)?.experiment_id
}));

export const createSaveJobPayload = (data: any) => {
  return {
    name: data.name,
    userId: LOGGEDIN_USER.name,
    designId: data.designId,
    priority: data.priority,
    backend: data.backend,
    maxRunTime: +data.maxRunTime,
    dataSpec: [data.dataSpec],
    modelSpec: {
      ...data.modelSpec,
      baseModel: {
        name: data.basemodelName,
        version: +data.basemodelVersion,
      },
      "dependencies": [
        "numpy >= 1.2.0"
      ],

    }
  }
}

export const createDatasetPayload = (datasets: any): DatasetPayload => {
  return Object.keys(datasets).reduce((acc, key) => ({
    role: datasets[key].role,
    datasetGroups: {
      ...(acc as unknown as any).datasetGroups || {},
      [key]: [...datasets[key].datasets.map((data: any) => data.value)]
    }
  }), {} as DatasetPayload);
}

export const mapDatasetsToSelectOption = (datasets: Dataset[] | undefined) => {
  return datasets?.map(dataset => ({
    label: `${dataset.name} - ${dataset.realm} - ${dataset.userId}`,
    value: dataset.id
  }));
}

export const getLabel = (label: string): string => {
  return `${label?.[0].toUpperCase()}${label?.substring(1)}`
}

export const createEntityKwargsPayload = (entity: string, groups: OptimizerGroup[]) => {
  return {
    sort: entity,
    kwargs: {
      ...groups.reduce((acc, group) => ({
        ...acc,
        [group.arg]: group.value
      }), {})
    }

  } as KwargPayload
}

export const mapJobToForm = (job: any) => {
  const datasets = getDatasetsFromJob(job);
  return {
    ...job,
    datasets
  };
};

const getDatasetsFromJob = (job: any) => {
  return job.dataSpec.map((data: any) => {
    const keys = Object.keys(data.datasetGroups);
    const mappedDatasets: any = {};
    for (const key of keys) {
      mappedDatasets[key] = {
        datasets: data.datasetGroups[key],
        role: data.role,
      }
    }

    return mappedDatasets;
  });
};

export const getHyperparametersFromJob = (job: any) => {
  return Object.keys(job.modelSpec.hyperparameters).map((key: string, index: number) => ({
    key,
    value: job.modelSpec.hyperparameters[key],
    id: index + 1,
  }))
}

export const getSelectorsFromJob = (job: any) => {
  return job.modelSpec.selector.sort;
}