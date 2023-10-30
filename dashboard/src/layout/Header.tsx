import { Image } from '@chakra-ui/react';
import { Box } from '@mui/material';
import avatarPlaceholder from '../assets/default-avatar.png';
import MenuOutlinedIcon from '@mui/icons-material/MenuOutlined';

const Header = () => {
  return (
    <Box
      padding="5px 20px"
      borderBottom="1px solid rgba(58, 53, 65, 0.12)" 
      height="50px"
      display="flex"
      alignItems="center"
      justifyContent="space-between"
    >
      <MenuOutlinedIcon fontSize="small" color="disabled" cursor="pointer"/>

      <Image
        src={avatarPlaceholder}
        height="30px"
        borderRadius="50%"
        cursor="pointer"
      />
    </Box>
  );
};

export default Header;
