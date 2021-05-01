mkdir -p %userprofile%/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > %userprofile%/.streamlit/config.toml
