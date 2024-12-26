function styleAllElementsWithTailwindAndDaisyUI() {
    // Define elements and their corresponding Tailwind + DaisyUI styles for a professional look
    const elements = [
        { selector: "body", classes: "antialiased min-h-screen" },
        { selector: "h1", classes: "text-4xl md:text-7xl font-bold text-secondary mb-8 pb-12 mt-12 py-12 font-bold underline" },
        { selector: "h2", classes: "text-3xl md:text-4xl font-bold text-secondary mb-2 mt-10 leading-snug" },
        { selector: "h3", classes: "text-2xl md:text-3xl font-medium mt-4 mb-3 leading-snug" },
        { selector: "h4", classes: "text-xl md:text-2xl font-medium mb-2 leading-snug" },
        { selector: "p", classes: "text-base md:text-lg leading-relaxed mb-4" },
        { selector: "button", classes: "btn btn-primary px-6 py-2 rounded-lg shadow-md hover:shadow-lg transition-all duration-300 ease-in-out" },
        { selector: "a", classes: "text-primary hover:text-primary-focus underline transition duration-300 ease-in-out" },
        { selector: "input", classes: "input input-bordered w-full max-w-md rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition duration-300 ease-in-out" },
        { selector: "select", classes: "select select-bordered w-full max-w-md rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition duration-300 ease-in-out" },
        { selector: "textarea", classes: "textarea textarea-bordered w-full max-w-md rounded-lg p-4 focus:ring-2 focus:ring-primary focus:border-primary transition duration-300 ease-in-out" },
        { selector: ".card", classes: "card bg-base-100 shadow-lg rounded-xl p-6 border border-gray-200 hover:shadow-xl transition-shadow duration-300 ease-in-out" },
        { selector: ".alert", classes: "alert shadow-md rounded-lg" },
        { selector: ".badge", classes: "badge badge-secondary text-xs px-3 py-1 rounded-full" },
        { selector: ".container", classes: "container mx-auto px-4 sm:px-6 lg:px-8" },
        { selector: ".grid", classes: "grid gap-6 md:grid-cols-2 lg:grid-cols-3" },
        { selector: ".flex", classes: "flex items-center justify-between" },
        { selector: ".section", classes: "py-12 md:py-16 lg:py-20" },
        { selector: "li", classes: "ml-10 list-disc" },
        { selector: "ul", classes: "mb-5 list-disc" },
        { selector: "pre", classes: "p-2 bg-base-200 rounded-lg" },
        { selector: "code", classes: "text-primary" },
        { selector: "img", classes: "border shadow-lg rounded-xl" },
        { selector: "figure", classes: "p-4" },
    ];

    // Apply the classes to all matching elements
    elements.forEach((item) => {
        const nodes = document.querySelectorAll(item.selector);
        nodes.forEach((node) => {
            node.classList.add(...item.classes.split(" "));
        });
    });
}


// Add structure for a table of contents and content layout
function setupLayout() {
    const body = document.querySelector("body");

    // Create main container
    const container = document.createElement("div");
    container.className = "flex flex-row h-screen";

    // Create table of contents column
    const toc = document.createElement("div");
    toc.className = "w-64  p-4 rounded-lg shadow-md fixed h-full overflow-y-auto hidden lg:block";
    toc.innerHTML = `
       <div id="index"></div>
    `;


    // Create content column
    const content = document.createElement("div");
    content.className = "flex-1  p-6 rounded-lg shadow-md ml-0 lg:ml-64 overflow-y-auto";
    content.innerHTML = `
       ${document.body.innerHTML}
    `;

    // Append TOC and content to the container
    container.appendChild(toc);
    container.appendChild(content);

    // Append container to the body
    body.innerHTML = ""; // Clear existing content
    body.appendChild(container);
}

// Execute the functions
styleAllElementsWithTailwindAndDaisyUI();
setupLayout();



function createTechnologyCategory(title, technologies) {
    return `
        <div class="m-0 p-0 flex align-content-center">
            <span class="text-sm font-semibold mb-4">${title}</span>
            <ul class="menu menu-horizontal w-full rounded-box">
                ${technologies.map(tech => createTechnologyItem(tech)).join('')}
            </ul>
        </div>
    `;
}

function createTechnologyItem(tech) {
    return `
        <li>
            <a class="flex items-center space-x-2">
                <img src="${tech.icon}" alt="${tech.name}" width="24" height="24" />
                <span>${tech.name}</span>
            </a>
        </li>
    `;
}

const webDevelopment = [
    { name: 'Python', icon: 'https://www.svgrepo.com/show/374016/python.svg' },
    { name: 'OpenCV', icon: 'https://www.svgrepo.com/show/354139/opencv.svg' },
    { name: 'Numpy', icon: 'https://www.svgrepo.com/show/354127/numpy.svg' },
];


document.getElementById("techs").innerHTML = `${createTechnologyCategory("Using:", webDevelopment)}`