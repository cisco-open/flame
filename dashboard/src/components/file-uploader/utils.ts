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

/**
 * Converting the file size to MB
 * @param size : Size to be converted;
 * @returns number
 *
 * @internal
 */
export const getFileSizeMB = (size: number): number => {
  return size / 1000 / 1000;
};

/**
 *
 * Check if the file uploaded is in the type list or not
 * @param file - The File uploaded
 * @param types - Available types
 * @returns boolean
 *
 * @internal
 */
export const checkType = (file: File, types: Array<string>): boolean => {
  const extension: string = file.name.split('.').pop() as string;
  const loweredTypes = types.map((type) => type.toLowerCase());
  return loweredTypes.includes(extension.toLowerCase());
};

/**
 * Get the files for input "accept" attribute
 * @param types - Allowed types
 * @returns string
 *
 * @internal
 */
export const acceptedExt = (types: Array<string> | undefined) => {
  if (types === undefined) return '';
  return types.map((type) => `.${type.toLowerCase()}`).join(',');
};