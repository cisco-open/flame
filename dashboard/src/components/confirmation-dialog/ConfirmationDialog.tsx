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

import { Modal, ModalOverlay, ModalContent, Text, ModalCloseButton, ModalBody, ModalFooter, Button } from "@chakra-ui/react"
import { isValid } from "date-fns"

interface Props {
  isOpen: boolean;
  actionButtonLabel: string;
  buttonColorScheme?: string;
  message: string;
  onClose: () => void;
  onAction: () => void;
}

const ConfirmationDialog = ({ isOpen, actionButtonLabel, message, buttonColorScheme = 'teal', onClose, onAction }: Props) => {
  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
    >
      <ModalOverlay />

      <ModalContent className="dataset-form">
        <ModalCloseButton />

        <ModalBody pb={6} display="flex" flexDirection="column" gap="10px">
          <Text>{message}</Text>
        </ModalBody>

        <ModalFooter>
          <Button onClick={onAction} size="xs" colorScheme={buttonColorScheme} mr={3} isDisabled={!isValid}>
            {actionButtonLabel}
          </Button>
          <Button onClick={onClose} size="xs">Cancel</Button>
        </ModalFooter>
      </ModalContent>
  </Modal>
  )
}

export default ConfirmationDialog