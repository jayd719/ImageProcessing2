function generateIndex(containerId = "index") {
    // Get all heading tags (h1 to h6)
    const headings = document.querySelectorAll("h2,h3, h4, h5, h6");
    if (!headings.length) {
        console.warn("No headings found in the document.");
        return;
    }

    // Create the container for the index
    let container = document.getElementById(containerId);
    if (!container) {
        container = document.createElement("div");
        container.id = containerId;
        document.body.insertBefore(container, document.body.firstChild);
    }
    container.innerHTML = `<h1 class="py-5 font-bold text-2xl text-secondary">${document.querySelector("h1").innerText}</h1>
    <h3 style='margin-top:15px;'>Table of Contents</h3>`;

    const indexList = document.createElement("ul");
    indexList.className = "menu"
    container.appendChild(indexList);

    headings.forEach(heading => {
        const level = parseInt(heading.tagName[1]);
        const listItem = document.createElement("li");

        if (!heading.id) {
            heading.id = `91oz.-${Math.random().toString(36).substr(2, 9)}`;
        }
        const link = document.createElement("a");
        link.href = `#${heading.id}`;
        link.className = "nav-link"
        link.textContent = heading.textContent;
        listItem.appendChild(link);
        listItem.style.marginLeft = `${(level - 1) * 10}px`;

        indexList.appendChild(listItem);
    });
}
generateIndex();

