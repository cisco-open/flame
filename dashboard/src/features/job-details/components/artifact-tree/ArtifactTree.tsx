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

// 1: Uncontrolled Tree
import { useRef } from "react";
import { Tree } from "react-arborist";
import ArtifactNode from "../artifact-node/ArtifactNode";
import './ArtifactTree.css';

interface Props {
  data: any;
}

const ArtifactTree = ({ data }: Props) => {
  const treeRef = useRef(null);

  return (
    <div className="artifact-tree-container">
      <Tree
        ref={treeRef}
        initialData={data}
        width={260}
        height={1000}
        indent={24}
        rowHeight={32}
        openByDefault={false}
      >
        {ArtifactNode}
      </Tree>
    </div>
  );
};

export default ArtifactTree;
