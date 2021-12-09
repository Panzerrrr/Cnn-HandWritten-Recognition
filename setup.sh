mkdir -p ~/.streamlit/echo "\
[general]\n\
email = \"nidalzaiani@gmail.com\"\n\
" > ~/.streamlit/credentials.nidal "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml