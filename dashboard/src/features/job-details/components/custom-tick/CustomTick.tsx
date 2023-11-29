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

const CustomTick = (data: any) => {
  if (typeof data.payload.value === 'string') {
    const lines = data.payload.value.split(' ');

    return (
      <g transform={`translate(${data.x},${data.y})`}>
          <text fontSize="10px" x={0} y={0} dy={16} fill="#666">
            <tspan textAnchor="middle" x="0">
              {lines[0]}
            </tspan>
            <tspan textAnchor="middle" x="0" dy="10">
              {lines[1]}
            </tspan>
          </text>
        </g>
    )
  }

  return (
    <g transform={`translate(${data.x},${data.y})`}>
        <text fontSize="10px" x={0} y={0} dy={16} fill="#666">
          <tspan textAnchor="middle" x="0">
            {data.payload.value}
          </tspan>
        </text>
      </g>
  )
}

export default CustomTick