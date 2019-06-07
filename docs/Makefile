# Install Markdown with
#
#	sudo apt-get install markdown
#
# or equivalent package manager.
# Install markdown-pdf with the Node Package Manager:
#
#	npm install -g markdown-pdf --unsafe-perm
#

muairss-gen-docs.html: muairss-gen-docs.md
	markdown muairss-gen-docs.md > muairss-gen-docs.html

muairss-gen-docs.pdf: muairss-gen-docs.md
	markdown-pdf muairss-gen-docs.md