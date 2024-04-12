import { Box, Icon, IconButton, Menu, MenuButton, Text, MenuItem, MenuList, Modal, ModalBody, ModalCloseButton, ModalContent, ModalHeader, ModalOverlay, Table, TableContainer, Tbody, Td, Th, Thead, Tr } from '@chakra-ui/react'
import { colors } from '@mui/material';
import React, { useEffect, useRef, useState } from 'react'
import { FaEllipsisVertical } from 'react-icons/fa6';
import { COLORS } from '../../../../constants';
import { Design } from '../../../../entities/Design';
import { getMappedLogs } from '../../utils';

interface Props {
  isOpen: boolean;
  log: string | undefined;
  taskName: string | undefined;
  onClose: () => void;
}

const COLUMNS = ['Time', 'File', 'Level', 'Thread', 'Function', 'Message'];

const TaskLogs = ({ isOpen, log, taskName, onClose }: Props) => {
  const initialRef: React.MutableRefObject<null> = useRef(null);
  const [ mappedLogs, setMappedLogs ] = useState<{value: string, id: number}[]>([]);

  useEffect(() => {
    setMappedLogs(getMappedLogs(log) || []);
  }, [log]);

  return (
    <Modal
      initialFocusRef={initialRef}
      isOpen={isOpen}
      onClose={onClose}
      size="full"
    >
      <ModalOverlay />

      <ModalContent height="100%" overflow="hidden" overflowY="auto">
        <ModalHeader textAlign="center">{ taskName?.toUpperCase() } LOGS</ModalHeader>

        <ModalCloseButton />

        <ModalBody display="flex" flexDirection="column" gap="20px">
          {
            mappedLogs?.[0]?.value?.split('|')?.length < 6 ?
              mappedLogs.map(log => <p key={log.id}>{ log.value}</p>) :

              <TableContainer flex={1} overflowY="auto" zIndex="1" backgroundColor={COLORS.offWhite} borderRadius="10px" padding="10px">
              <Table variant='simple' fontSize={12} size="sm">
                <Thead>
                    <Tr>
                        {COLUMNS.map(column => <Th key={column}>{column}</Th>)}
                    </Tr>
                </Thead>

                <Tbody>
                    {
                      mappedLogs?.map((log) =>
                        <Tr height="50px" key={log.id}>
                            {
                              log.value.split('|').map((l, index) =>
                                <Td padding="10px 20px">
                                  <Text key={`${index}${log.id}`}>{l}</Text>
                                </Td>
                              )
                            }
                        </Tr>
                      )
                    }
                </Tbody>
              </Table>
            </TableContainer>
          }

        </ModalBody>
      </ModalContent>
    </Modal>
  )
}

export default TaskLogs