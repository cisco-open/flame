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

import { Box, SimpleGrid, FormControl, Text } from '@chakra-ui/react';
import { MultiSelect, Option } from 'chakra-multiselect';
import { useEffect, useState } from 'react';
import { Dataset } from '../../../../entities/Dataset';
import { DatasetPayload } from '../../../../entities/Job';
import { createDatasetPayload, mapDatasetsToSelectOption } from '../../utils';
import './DatasetForm.css';

export interface SelectOption {
  label: string,
  value: string,
}

interface Props {
	datasetControls: { label: string, controls: string[]}[] | undefined;
	datasets: Dataset[] | undefined;
	mappedJobToForm: any;
	selectedDesignId: string;
	setDatasetPayload: (payload: DatasetPayload) => void;
}

const DatasetForm = ({ datasetControls, datasets, mappedJobToForm, selectedDesignId, setDatasetPayload }: Props) => {
	const [ mappedDatasets, setMappedDatasets ] = useState<Option[] | undefined>(undefined);
	const [ selectedDatasets, setSelectedDatasets ] = useState<any>(null);
	const [filteredDatasets, setFilteredDatasets] = useState<Dataset[] | undefined>([]);

	useEffect(() => {
	  setFilteredDatasets(datasets);
	}, [datasets]);

	useEffect(() => {
		if (!mappedJobToForm || mappedJobToForm.designId !== selectedDesignId) { return; }
		// @TODO -> fix datasets to handle a list of datasets
		const jobDatasets = mappedJobToForm.datasets.map((datasetGroup: any) => {
			const mappedGroup = Object.keys(datasetGroup).reduce((acc: any, key: string) => ({
				...acc,
				[key]: {
					...datasetGroup[key],
					datasets: datasetGroup[key].datasets?.map((dataset: string) => ({
						label: datasets?.find(d => d.id === dataset)?.name || '',
						value: dataset,
					}))
				}
			}), {});

			return mappedGroup;
		});
		// @TODO - fix multiple datasets
		setSelectedDatasets(jobDatasets[0]);
	}, [mappedJobToForm])

	useEffect(() => {
		if (!selectedDatasets) { return; }
		setDatasetPayload(createDatasetPayload(selectedDatasets));
	}, [selectedDatasets])

	const handleDatasetChange = (event: any, dataset: any, control: any) => {
		setSelectedDatasets({ ...selectedDatasets, [control]: { datasets: [...event], role: dataset.label }});
	}

	useEffect(() => {
		if (!filteredDatasets?.length) { return; }
		setMappedDatasets(mapDatasetsToSelectOption(filteredDatasets));
	}, [filteredDatasets])

	return (
		<Box display="flex" flexDirection="column" gap="20px" height="100%" overflow="hidden" overflowY="auto" alignItems="center" padding="10px">
			<Text as="h4" textAlign="center">Select datasets for data consumer roles</Text>

			<SimpleGrid
				columns={1}
				spacing="20px"
				borderRadius="4px"
				padding="10px"
				width="50%"
			>
				<Box
					borderRadius="4px"
					padding="10px"
					display="flex"
					flexDirection="column"
					gap="20px"
					alignItems="center"
				>
					{
						datasetControls?.map(dataset =>
							<FormControl key={dataset.label} display="flex" flexDirection="column" gap="20px">
								{
									dataset.controls.map(control =>
										<FormControl key={control} display="flex" flexDirection="column" gap="10px">
											<Text>Data consumer role: <strong>{dataset?.label}</strong>, Group: <strong>{control}</strong></Text>
											<Text fontSize="10px" fontStyle="italic">Dataset name format: Name, Realm, User ID</Text>
											<MultiSelect
												options={mappedDatasets}
												value={selectedDatasets?.[control]?.datasets}
												onChange={(event) => handleDatasetChange(event, dataset, control)}
											/>
										</FormControl>
									)
								}
							</FormControl>
						)
					}
				</Box>
			</SimpleGrid>
		</Box>
	)
}

export default DatasetForm;