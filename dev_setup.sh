pip3 install --user -r requirements.txt
cat chrome-patch.txt ~/.local/lib/python3.10/site-packages/chromadb/__init__.py > ~/.local/lib/python3.10/site-packages/chromadb/__init__.py.tmp
mv ~/.local/lib/python3.10/site-packages/chromadb/__init__.py.tmp ~/.local/lib/python3.10/site-packages/chromadb/__init__.py
