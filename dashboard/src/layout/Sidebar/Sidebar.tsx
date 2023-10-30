import { VStack, Image, Text } from '@chakra-ui/react';
import { NavLink, useNavigate } from 'react-router-dom';
import logo from '../../assets/flame-logo.png';
import './Sidebar.css';
import menuItems from '../../menu-items';

const Sidebar = () => {
  const navigate = useNavigate();
  return (
    <VStack
      className="sidebar"
      alignItems="flex-start"
      borderRight='1px solid rgba(58, 53, 65, 0.12)'
      gap="5px"
      padding="10px 5px 10px 0"
      height="100%"
      boxShadow="rgba(58, 53, 65, 0.42) 0px 4px 8px -4px"
    >
      <Image
        paddingLeft="5px"
        height='50px'
        src={logo}
        marginBottom="-5px"
        onClick={() => navigate('/')}
        cursor='pointer'
      />

      {menuItems.map(item =>
        <NavLink to={item.url} key={item.id} className="sidebar-link">
          {item.icon}

          <p className="item-title">{item.title}</p>
        </NavLink>
      )}
    </VStack>
  )
}

export default Sidebar