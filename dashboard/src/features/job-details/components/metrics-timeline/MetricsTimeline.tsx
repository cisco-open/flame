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

import React, { useEffect, useState } from 'react';
import 'react-calendar-timeline/lib/Timeline.css'
import Timeline, {
  TimelineHeaders,
  DateHeader,
} from 'react-calendar-timeline/lib';
import 'react-calendar-timeline/lib/Timeline.css'
import { Box, Text } from '@chakra-ui/react';
import { MetricsTimelineData, TimelineGroup, Timelineitem, MetricsTimelineDataItem } from '../../types';
import { colors } from '../../../..';
import WorkerDetails from '../worker-details/WorkerDetails';
import './MetricsTimeline.css';

const keys = {
  groupIdKey: "id",
  groupTitleKey: "title",
  itemIdKey: "id",
  itemTitleKey: "title",
  itemDivTitleKey: "title",
  itemGroupKey: "group",
  itemTimeStartKey: "start",
  itemTimeEndKey: "end",
  groupLabelKey: "title",
};

interface Props {
  data: MetricsTimelineData;
  runs: any;
}

const itemColors = [
  '#1D1B31', '#4895EF', '#232946', '#4CC9F0', '#3B2E5A', '#4CC9F0', '#A12059', '#3E0568', '#7209B7', '#560BAD',
  '#480CA8', '#3A0CA3', '#3F37C9', '#4361EE', '#4CC9F0', '#45B69C', '#3D9880', '#396B5E', '#264653',
  '#2A9D8F', '#2ECC71', '#ABEBC6', '#F0F3BD', '#F9C74F', '#F8961E', '#F3722C', '#F94144', '#7209B7',
  '#560BAD', '#480CA8', '#3A0CA3', '#3F37C9', '#4361EE', '#4895EF', '#45B69C', '#3D9880', '#396B5E',
  '#264653', '#2A9D8F', '#2ECC71', '#ABEBC6', '#F0F3BD', '#F9C74F', '#F8961E', '#F3722C', '#F94144',
  '#7209B7', '#560BAD', '#480CA8', '#3A0CA3', '#3F37C9', '#4361EE', '#4895EF', '#4CC9F0', '#45B69C', '#3D9880',
  '#396B5E', '#264653', '#2A9D8F', '#2ECC71', '#ABEBC6', '#F0F3BD', '#F9C74F', '#F8961E', '#F3722C', '#F94144',
  '#7209B7', '#560BAD', '#480CA8', '#3A0CA3', '#3F37C9', '#4361EE', '#4895EF', '#4CC9F0', '#45B69C',
  '#3D9880', '#396B5E', '#264653', '#2A9D8F', '#2ECC71', '#ABEBC6', '#F0F3BD', '#F9C74F', '#F8961E', '#F3722C',
  '#F94144',
];

const MetricsTimeline = ({ data, runs }: Props) => {
  const [min, setMin] = useState<Date>();
  const [max, setMax] = useState<Date>();
  const [groups, setGroups] = useState<TimelineGroup[]>([]);
  const [items, setItems] = useState<Timelineitem[]>([]);
  const [selectedWorker, setSelectedWorker] = useState<string>();
  const [runDetails, setRunDetails] = useState<string>();

  useEffect(() => {
    const selectedRun = runs?.find((run: any) => selectedWorker?.includes(run.taskId));
    setRunDetails({ ...selectedRun });
  }, [runs, selectedWorker]);

  useEffect(() => {
    if (!data || (data && !Object.keys(data)?.length)) { return; }

    const keyToColorMap: { [key: string]: string } = {};

    const groups: TimelineGroup[] = Object.keys(data).map((key, index) => ({
      id: `${key}-${index}`,
      title: <Box onClick={() => onGroupClick(key)} cursor="pointer">
        <Text as="p" textDecoration="underline" color={colors.primary.normal}>{key}</Text>
      </Box>,
      name: key,
    }));

    const items: Timelineitem[] = Object.keys(data)
      .reduce((acc: MetricsTimelineDataItem[], key: string) => [...acc, ...data[key]], [])
      .map((item: MetricsTimelineDataItem, index: number) => {
        if (!(item.key in keyToColorMap)) {
          keyToColorMap[item.key] = itemColors[Object.keys(keyToColorMap).length + 1]; // Assign a new color if key is encountered for the first time
        }

        const group = groups?.find((group) => group.name === item.category)?.id || ''
        return {
          id: `${item.category}-${item.key}-${item.step}`,
          group,
          title: item.key.split('.')[0],
          start: item.start,
          end: item.start + (Math.round(item.value) * 1000),
          bgColor: keyToColorMap[item.key]
        }
      });

    const min = new Date(Math.min(...items.map(item => item.start)));
    const max = new Date(Math.max(...items.map(item => item.end)));

    setMin(min);
    setMax(max);
    setGroups(groups);
    setItems(items);
  }, [data]);

  const onGroupClick = (worker: string) => {
    setSelectedWorker(undefined);

    setTimeout(() => {
      setSelectedWorker(worker);
    }, 100)
  }

  const itemRenderer = ({ item, itemContext, getItemProps }: any) => {
    const backgroundColor = item.bgColor;
    return (
      <div
        {...getItemProps({
          style: {
            backgroundColor,
            borderStyle: "solid",
            borderWidth: 1,
            borderRadius: 4,
            borderLeftWidth: itemContext.selected ? 3 : 1,
            borderRightWidth: itemContext.selected ? 3 : 1
          },
        })}
      >
        <div
          style={{
            height: itemContext.dimensions.height,
            overflow: "hidden",
            paddingLeft: 3,
            textOverflow: "ellipsis",
            whiteSpace: "nowrap"
          }}
        >
          {itemContext.title}
        </div>
      </div>
    );
  };



  return (
    <Box display="flex" flexDirection="column" gap="20px">
      {
        !!min && !!max && !!items.length && !!groups.length &&
        <Timeline
          groups={groups}
          items={items}
          defaultTimeStart={min}
          defaultTimeEnd={max}
          minZoom={100}
          keys={keys}
          itemRenderer={itemRenderer}
        >
          <TimelineHeaders className="sticky">
            <DateHeader unit="minute" />
          </TimelineHeaders>
        </Timeline>
      }

      {
        selectedWorker &&
        <Box display="flex" flexDirection="column" gap="20px">
          <Text as="h3" fontWeight="bold" textAlign="center">{selectedWorker}</Text>

          <WorkerDetails runDetails={runDetails}/>
        </Box>
      }
    </Box>
  );
};

export default MetricsTimeline;