/**
 * Copyright 2023 Cisco Systems, Inc. and its affiliates
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { Metric, Run, RunResponse } from "../../entities/JobDetails";
import { Task } from "../../entities/Task";
import { UiMetric } from "./components/job-run-timeline/JobRunTimeline";
import { MappedTask } from "./types";

export const getEdges = (tasks: Task[]) => {
  const pairs = [];
  const mappedTasks = tasks.map((task: Task) => ({
    ...task,
    group: Object.values(task?.groupAssociation || {}) as string[],
    count: tasks.filter((t: Task) => t.role === task.role).length
  })).sort(((a: MappedTask, b: MappedTask) => a.count - b.count));

  for (let i = 0; i < mappedTasks.length; i++) {
    const pair = mappedTasks.find((task, index) =>
      task.taskId !== mappedTasks[i].taskId &&
      task.role !== mappedTasks[i].role &&
      i >= index &&
      hasSameGroup(task, mappedTasks[i])
    );

    if (pair) {
      pairs.push({
        first: pair,
        second: mappedTasks[i],
      });
    }
  }
  return pairs.map(({ first, second }) => ({
    id: `${first.taskId}-${second.taskId}`,
    source: first.taskId,
    target: second.taskId,
    type: 'floating',
    label: `${getEdgeLabel(first, second)}`,
  }))
}

const getEdgeLabel = (first: Task, second: Task) => {
  return first.group?.find((group) => second.group?.includes(group));
}

const hasSameGroup = (task: Task, secondTask: Task) => {
  return !!task.group?.find((group: string) => secondTask.group?.includes(group));
}

export const getRuntimeMetrics = (runs: Run[] | undefined) => {
  const names: string[] = [];

  const metrics = (runs || []).map(run => {
    const runName = getRunName(run);
    names.push(runName);

    const startTimeMetrics = run.data.metrics
      .filter(metric => metric.key.includes('starttime'))
      .sort((m1, m2) => m1.timestamp - m2.timestamp)

    const runTimeMetrics = run.data.metrics
      .filter(metric => metric.key
      .includes('runtime')).sort((m1, m2) => m1.timestamp - m2.timestamp)

    return {
      name: runName,
      values: [...startTimeMetrics, ...runTimeMetrics]
        ?.map((metric, i) => {
          return {
            name: metric.key,
            category: runName,
            runId: run.info.run_id,
            order: i,
          }
        })
    }
  }).map(entry => entry.values).reduce((acc, item) => acc = [...acc, ...(item || [])], []);

  return {
    metrics,
    names,
  }
}

export const getRunName = (run: Run): string => {
  return run?.data?.tags?.find(tag => tag.key.includes('runName'))?.value || '';
}

export const getNodes = (tasks: Task[]) => {
  return tasks.map((task) => {
    return {
      id: task.taskId,
      data: {
        label: task.role,
        id: task.taskId,
        status: task.state,
      },
      position: { x: 0, y: 0 },
      type: 'customNodeNoInteraction',
    }
  });
}

export const getTasksWithLevelsAndCounts = (tasks: Task[]): Task[] => {
  const tasksWithCount = tasks?.map(task => ({
    ...task,
    count: tasks.filter((t: any) => t.role === task.role).length,
  }));
  const levels = tasksWithCount.map((task: any) => task.count);
  const counts = Array.from(new Set(levels)).sort((a: number, b: number) => a - b);
  return tasksWithCount.map((task: any) => ({
    ...task,
    level: counts.indexOf(task.count) + 1,
    group: Object.values(task?.groupAssociation || {}),
  })).sort((a: any, b: any) => {
    const aSorted = a.group.sort((a: any, b: any) => a - b).join('');
    const bSorted = b.group.sort((a: any, b: any) => a - b).join('');
    if ( aSorted < bSorted ){
      return -1;
    }
    if ( aSorted > bSorted ){
      return 1;
    }
    return 0;
  });
}

export const sortMetrics = (metrics: Metric[]) => metrics?.sort((a: Metric, b: Metric) => {
  if ( a.key < b.key ){
    return -1;
  }
  if ( a.key > b.key ){
    return 1;
  }
  return 0;
});

export const getLatestRuns = (runsResponse: RunResponse | undefined, tasks: Task[] | undefined) => {
  const taskIds = tasks?.map(task => task.taskId.substring(0, 8));

  return runsResponse?.runs.filter((run, index, runs) => {
    const runName = getRunName(run);
    return run.info.start_time &&
      run.info.end_time &&
      taskIds?.includes(runName.substring(runName.length - 8)) &&
      hasTheGreatestTimestamp(run, runs)
  }
  )
}

const hasTheGreatestTimestamp = (run: Run, runs: Run[]) => {
  const targetRuns = runs.filter(r => getRunName(r) === getRunName(run) && run.info.start_time && run.info.end_time);
  let targetRun = targetRuns[0];
  for (let run of targetRuns) {
    if (targetRun.info.start_time < run.info.start_time) {
      targetRun = run;
    }
  }

  return targetRun.info.run_id === run.info.run_id;
}

export const mapMetricResponse = (res: any, list: any, metric: UiMetric) => {
  const { category } = metric;

  return (res as unknown as any).metrics.map((m: Metric) => {
    const startTimeCounterpart = list[category]
      .find((metric: Metric) => metric.key.includes(m.key.split('.')[0]) &&
        metric.key.includes('starttime') &&
        m.step === metric.step
      );
    const start = Math.round((startTimeCounterpart?.value || 1) * 1000);
    return {
      ...m,
      category,
      start,
    }
  })
}
