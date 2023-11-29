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

import styled, { css } from 'styled-components';

const primary = '#0658c2',
  darkGray = '#666',
  lightGray = '#999';

const defaultStyle = css`
  display: flex;
  align-items: center;
  width: 100%;
  height: 30vh;
  border: dashed 2px ${primary};
  padding: 8px 16px 8px 8px;
  border-radius: 5px;
  cursor: pointer;
  flex-grow: 0;

  &.is-disabled {
    border: dashed 2px ${darkGray};
    cursor: no-drop;
    svg {
      fill: ${darkGray};
      color: ${darkGray};
      path {
        fill: ${darkGray};
        color: ${darkGray};
      }
    }
  }
`;
export const UploaderWrapper = styled.label<any>`
  position: relative;
  ${(props: any) => (props.overRide ? '' : defaultStyle)};
  &:focus-within {
    outline: 2px solid black;
  }
  & > input {
    display: block;
    opacity: 0;
    position: absolute;
    pointer-events: none;
    width: 100%;
  }
`;
/**
 *
 * @internal
 */
export const HoverMsg = styled.div`
  border: dashed 2px ${darkGray};
  border-radius: 5px;
  background-color: ${lightGray};
  opacity: 0.5;
  position: absolute;
  top: 0;
  right: 0;
  left: 0;
  bottom: 0;
  & > span {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translateX(-50%) translateY(-50%);
  }
`;
/**
 *
 * @internal
 */
export const DescriptionWrapper = styled.div<{ error: boolean }>`
  display: flex;
  justify-content: space-between;
  flex-grow: 1;
  & > span {
    font-size: 12px;
    color: ${(props: any) => (props.error ? 'red' : darkGray)};
  }
  .file-types {
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
    max-width: 100px;
  }
`;
/**
 *
 * @internal
 */
export const Description = styled.span`
  font-size: 14px;
  color: ${darkGray};
  span {
    text-decoration: underline;
  }
`;
