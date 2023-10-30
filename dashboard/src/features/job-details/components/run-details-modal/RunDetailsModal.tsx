import { Modal, ModalOverlay, ModalContent, ModalHeader, ModalCloseButton, ModalBody, Tab, TabList, TabPanel, TabPanels, Tabs } from '@chakra-ui/react';
import React, { useRef } from 'react';
import RunMetrics from '../run-metrics/RunMetrics';
import RunModelArtefact from '../run-model-artefact/RunModelArtefact';
import RunParameters from '../run-parameters/RunParameters';
import './RunDetailsModal.css';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  runDetails: any;
}

const RunDetailsModal = ({ isOpen, onClose, runDetails }: Props) => {
  const initialRef: React.MutableRefObject<null> = useRef(null);

  const handleClose = () => {
    onClose();
  }

  return (
    <Modal
      initialFocusRef={initialRef}
      isOpen={isOpen}
      onClose={handleClose}
      size="5xl"
    >
      <ModalOverlay />

      <ModalContent className="run-details-content">
        <ModalHeader textAlign="center">{runDetails?.info?.run_name}</ModalHeader>

        <ModalCloseButton />

        <ModalBody pb={6} display="flex" flexDirection="column" gap="10px">
          <Tabs className="run-details-tabs">
            <TabList>
              <Tab fontSize="12px">Hyperparameters</Tab>
              <Tab fontSize="12px">Metrics</Tab>
              <Tab fontSize="12px">Model Artifact</Tab>
            </TabList>

            <TabPanels className="run-details-tab-panels">
              <TabPanel className="run-details-tab-panel">
                <RunParameters parameters={runDetails?.data?.params} />
              </TabPanel>
              <TabPanel className="run-details-tab-panel">
                <RunMetrics metrics={runDetails?.data?.metrics} run={runDetails}/>
              </TabPanel>
              <TabPanel className="run-details-tab-panel">
                <RunModelArtefact />
              </TabPanel>
            </TabPanels>
          </Tabs>
        </ModalBody>
      </ModalContent>
    </Modal>
  )
}

export default RunDetailsModal