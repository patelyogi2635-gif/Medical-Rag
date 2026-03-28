async function sendQuery() {
    const input = document.getElementById("query");
    const query = input.value;

    const chatBox = document.getElementById("chat-box");

    // User message
    chatBox.innerHTML += `<div class="message user">${query}</div>`;

    // Loader
    const loaderId = "loading-" + Date.now();
    chatBox.innerHTML += `<div class="message bot" id="${loaderId}">🧠 Thinking...</div>`;

    chatBox.scrollTop = chatBox.scrollHeight;

    input.value = "";

    try {
        const res = await fetch("http://localhost:8000/query", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ query })
        });

        const data = await res.json();

        document.getElementById(loaderId).remove();

        chatBox.innerHTML += `
            <div class="message bot">
                ${data.answer}
                <div class="verified">✅ Verified: ${data.verified}</div>
            </div>
        `;

    } catch (err) {
        document.getElementById(loaderId).innerText = "❌ Server error";
    }

    chatBox.scrollTop = chatBox.scrollHeight;
}