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