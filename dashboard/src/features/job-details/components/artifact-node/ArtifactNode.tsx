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

import { useContext } from "react";
import { AiFillFolder, AiFillFile } from "react-icons/ai";
import { MdArrowRight, MdArrowDropDown } from "react-icons/md";
import './ArtifactNode.css';
import { ArtifactContext } from "../../ArtifactContext";

interface Props {
  node?: any;
  style?: any;
  dragHandle?: any;
  tree?: any;
}

const ArtifactNode = ({ node, style, dragHandle, tree }: Props) => {
  const CustomIcon = node?.data?.icon;
  const iconColor = node?.data?.iconColor;
  const { onFileSelect } = useContext(ArtifactContext);

  return (
    <div
      className={`node-container ${node.state.isSelected ? "isSelected" : ""}`}
      style={style}
      ref={dragHandle}
    >
      <div
        className="node-content"
        onClick={() => { onFileSelect(node); return node.isInternal && node.toggle() }}
      >
        {node.isLeaf ? (
          <>
            <span className="arrow"></span>
            <span className="file-folder-icon">
              {CustomIcon ? (
                <CustomIcon color={iconColor ? iconColor : "#6bc7f6"} />
              ) : (
                <AiFillFile color="#6bc7f6" />
              )}
            </span>
          </>
        ) : (
          <>
            <span className="arrow">
              {node.isOpen ? <MdArrowDropDown /> : <MdArrowRight />}
            </span>
            <span className="file-folder-icon">
              {CustomIcon ? (
                <CustomIcon color={iconColor ? iconColor : "#f6cf60"} />
              ) : (
                <AiFillFolder color="#f6cf60" />
              )}
            </span>
          </>
        )}
        <span className="node-text">
          {node.isEditing ? (
            <input
              type="text"
              defaultValue={node.data.name}
              onFocus={(e) => e.currentTarget.select()}
              onBlur={() => node.reset()}
              onKeyDown={(e) => {
                if (e.key === "Escape") node.reset();
                if (e.key === "Enter") node.submit(e.currentTarget.value);
              }}
              autoFocus
            />
          ) : (
            <span>{node.data.name}</span>
          )}
        </span>
      </div>
    </div>
  );
};

export default ArtifactNode;
