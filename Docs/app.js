function styleAllElementsWithTailwind() {
    // Apply modern, professional Tailwind CSS classes to elements dynamically
    const elements = [
        { selector: "body", classes: "bg-gray-100 font-sans text-gray-800 antialiased" },
        { selector: "h1", classes: "text-4xl font-bold text-gray-900 text-center mb-6 leading-tight" },
        { selector: "h2", classes: "text-3xl font-semibold text-gray-800 text-center mb-4 leading-snug" },
        { selector: "h3", classes: "text-2xl font-medium text-gray-700 mb-3 leading-snug" },
        { selector: "h4", classes: "text-xl font-medium text-gray-600 mb-2 leading-snug" },
        { selector: "h5", classes: "text-lg font-medium text-gray-500 mb-1 leading-snug" },
        { selector: "h6", classes: "text-base font-semibold text-gray-400 leading-snug" },
        { selector: "p", classes: "text-base leading-relaxed mb-4 text-gray-700" },
        { selector: "div", classes: "bg-white shadow-md rounded-lg p-6 mb-8 max-w-4xl mx-auto" },
        { selector: "button", classes: "bg-blue-600 text-white px-5 py-2 rounded-md hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 transition-all shadow-sm" },
        { selector: "a", classes: "text-blue-600 hover:text-blue-700 underline transition duration-150" },
    ];

    // Apply the classes to all matching elements
    elements.forEach((item) => {
        const nodes = document.querySelectorAll(item.selector);
        nodes.forEach((node) => {
            node.classList.add(...item.classes.split(" "));
        });
    });
}

// Execute the function to style all elements
styleAllElementsWithTailwind();
