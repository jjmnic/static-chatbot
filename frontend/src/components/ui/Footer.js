import { Box, Typography } from "@mui/material";

function Footer() {
  return (
    <Box
      component="footer"
      sx={{
        padding: "16px",
        marginTop: "50px",
      }}
    >
      <Typography variant="body2" color="textSecondary" textAlign="center">
        &copy; 2025 Jal Jeevan Mission. All rights reserved.
      </Typography>
    </Box>
  );
}

export default Footer;