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

import { Run } from "../../entities/JobDetails";
import { Task } from "../../entities/Task";

export const getEdges = (tasks: Task[]) => {
  const pairs = [];
  const mappedTasks = tasks.map((task: Task) => ({
    ...task,
    group: Object.values(task?.groupAssociation || {}) as string[],
    count: tasks.filter((t: Task) => t.role === task.role).length
  })).sort(((a: any, b: any) => a.count - b.count));

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

export const getInitialFileStructure = (data: any) => {
  return data?.files?.map(((file: any, index: number) => ({
    id: `${index}`,
    name: file.path,
    children: [],
    isDir: true,
  })));
}

export const getFileStructure = (artifacts: any, fileStructure: any) => {
  const fs = [...fileStructure];
  artifacts?.files?.map((file: any) => {
    const parentIndex = fs.findIndex((f: any) => f.name === file.path.split('/')[0]);
    const parent = fs[parentIndex];
    const fileName = file.path.split(`${parent.name}/`)[1];

    if (file.is_dir) {
      fs[parentIndex].children.push({
        id: `${parent.id}-${fileName}`,
        name: fileName,
        parentName: parent.name,
        children: [],
        path: file.path,
        isDir: file.is_dir
      })
    } else {
      fs[parentIndex].children.push({
        id: `${parent.id}-${fileName}`,
        parentName: parent.name,
        name: fileName,
        path: file.path,
        isDir: file.is_dir
      })
    }

  });

  return fs;
}

export const getRuntimeMetrics = (runs: Run[] | undefined) => {
  const names: string[] = [];

  const runtimes = (runs || []).map(run => {
    names.push(run.info.run_name);

    return {
      name: run.info.run_name,
      values: run.data.metrics
        ?.filter(metric => metric.key.includes('runtime') || metric.key.includes('starttime'))
        ?.map((metric) => {
          return {
            name: metric.key,
            category: run.info.run_name,
            runId: run.info.run_id,
          }
        }).sort((a, b) => {
          if (a.name < b.name) {
            return -1;
          } else if (a.name > b.name) {
            return 1;
          }
          return 0;
        })
    }
  }).map(entry => entry.values).reduce((acc, item) => acc = [...acc, ...(item || [])], []);

  return {
    runtimes,
    names,
  }
}

export const getNodes = (tasks: Task[], runs: Run[] | undefined) => {
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
  const counts = Array.from(new Set(levels)).sort((a: any, b: any) => a - b);
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

export const sortMetrics = (metrics: any) => metrics?.sort((a: any, b: any) => {
  if ( a.key < b.key ){
    return -1;
  }
  if ( a.key > b.key ){
    return 1;
  }
  return 0;
});
