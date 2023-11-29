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

import { useState, useEffect, useCallback } from 'react';

let draggingCount = 0;
type Params = {
  labelRef: any;
  inputRef: any;
  multiple?: boolean | false;
  handleChanges: (arg0: Array<File>) => void;
  onDrop?: (arg0: Array<File>) => void;
};

/**
 *
 * @param data - labelRef, inputRef, handleChanges, onDrop
 * @returns boolean - the state.
 *
 * @internal
 */
export default function useDragging({
  labelRef,
  inputRef,
  multiple,
  handleChanges,
  onDrop
}: Params): boolean {
  const [dragging, setDragging] = useState(false);
  const handleClick = useCallback(() => {
    inputRef.current.click();
  }, [inputRef]);

  const handleDragIn = useCallback((ev: any) => {
    ev.preventDefault();
    ev.stopPropagation();
    draggingCount++;
    if (ev.dataTransfer.items && ev.dataTransfer.items.length !== 0) {
      setDragging(true);
    }
  }, []);
  const handleDragOut = useCallback((ev: any) => {
    ev.preventDefault();
    ev.stopPropagation();
    draggingCount--;
    if (draggingCount > 0) return;
    setDragging(false);
  }, []);
  const handleDrag = useCallback((ev: any) => {
    ev.preventDefault();
    ev.stopPropagation();
  }, []);
  const handleDrop = useCallback(
    (ev: any) => {
      ev.preventDefault();
      ev.stopPropagation();
      setDragging(false);
      draggingCount = 0;

      const eventFiles = ev.dataTransfer.files;
      if (eventFiles && eventFiles.length > 0) {
        const files = multiple ? eventFiles : eventFiles[0];
        const success = handleChanges(files);
        if (onDrop) onDrop(files);
      }
    },
    [handleChanges]
  );
  useEffect(() => {
    const ele = labelRef.current;
    ele.addEventListener('click', handleClick);
    ele.addEventListener('dragenter', handleDragIn);
    ele.addEventListener('dragleave', handleDragOut);
    ele.addEventListener('dragover', handleDrag);
    ele.addEventListener('drop', handleDrop);
    return () => {
      ele.removeEventListener('click', handleClick);
      ele.removeEventListener('dragenter', handleDragIn);
      ele.removeEventListener('dragleave', handleDragOut);
      ele.removeEventListener('dragover', handleDrag);
      ele.removeEventListener('drop', handleDrop);
    };
  }, [
    handleClick,
    handleDragIn,
    handleDragOut,
    handleDrag,
    handleDrop,
    labelRef
  ]);

  return dragging;
}