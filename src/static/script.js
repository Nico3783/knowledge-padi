// Select DOM elements
const chatbotToggler = document.querySelector(".chatbot-toggler");
const closeBtn = document.querySelector(".close-btn");
const chatbox = document.querySelector(".chatbox");
const chatInput = document.querySelector(".chat-input textarea");
const sendChatBtn = document.querySelector("#send-btn");
const preprocessBtn = document.querySelector("#preprocess-btn"); // Preprocess trigger button

// Initialize user message variable
let userMessage = null;

// Default chatbox height for input auto-resizing
const inputInitHeight = chatInput.scrollHeight;

// Flask API endpoints
const QUERY_API_URL = "/api/query";
const PREPROCESS_API_URL = "/preprocess";

// Function to create chat message elements
const createChatLi = (message, className) => {
  const chatLi = document.createElement("li");
  chatLi.classList.add("chat", className);

  let chatContent =
    className === "outgoing"
      ? `<p></p>` // User message
      : `<span class="material-symbols-outlined">smart_toy</span><p></p>`; // Bot message

  chatLi.innerHTML = chatContent;
  chatLi.querySelector("p").textContent = message;
  return chatLi;
};

// Function to generate a response from the RAG pipeline
const generateResponse = async (chatElement) => {
  const messageElement = chatElement.querySelector("p");

  // Prepare POST request options
  const requestOptions = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: userMessage }), // Send user message as "query"
  };

  try {
    // Send request to Flask backend
    const response = await fetch(QUERY_API_URL, requestOptions);
    const data = await response.json();

    if (!response.ok) throw new Error(data.error || "Something went wrong");

    // Display chatbot response
    messageElement.textContent = data.response;
  } catch (error) {
    // Display error message
    messageElement.classList.add("error");
    messageElement.textContent = error.message;
  } finally {
    chatbox.scrollTo(0, chatbox.scrollHeight);
  }
};

// Function to handle user input
const handleChat = () => {
  userMessage = chatInput.value.trim(); // Get user message
  if (!userMessage) return;

  // Clear input and adjust its height
  chatInput.value = "";
  chatInput.style.height = `${inputInitHeight}px`;

  // Add user message to chatbox
  chatbox.appendChild(createChatLi(userMessage, "outgoing"));
  chatbox.scrollTo(0, chatbox.scrollHeight);

  // Display "Thinking..." message while awaiting response
  setTimeout(() => {
    const incomingChatLi = createChatLi("...", "incoming");
    chatbox.appendChild(incomingChatLi);
    chatbox.scrollTo(0, chatbox.scrollHeight);
    generateResponse(incomingChatLi);
  }, 600);
};

// Function to preprocess documents
const preprocessDocuments = async () => {
  preprocessBtn.textContent = "Processing..."; // Show loading state
  preprocessBtn.disabled = true; // Disable the button to prevent duplicate requests

  try {
    // Send request to the preprocessing endpoint
    const response = await fetch(PREPROCESS_API_URL, { method: "POST" });
    const data = await response.json();

    if (!response.ok) throw new Error(data.error || "Preprocessing failed");

    alert("Documents preprocessed and indexed successfully!");
  } catch (error) {
    alert(`Error: ${error.message}`);
  } finally {
    preprocessBtn.textContent = "Preprocess Documents";
    preprocessBtn.disabled = false;
  }
};

// Adjust input height dynamically
chatInput.addEventListener("input", () => {
  chatInput.style.height = `${inputInitHeight}px`;
  chatInput.style.height = `${chatInput.scrollHeight}px`;
});

// Handle "Enter" key for input submission
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleChat();
  }
});

// Handle send button click
sendChatBtn.addEventListener("click", handleChat);

// Handle toggling of chatbot visibility
chatbotToggler?.addEventListener("click", () => {
  document.body.classList.toggle("show-chatbot");
});

// Handle closing the chatbot (for widget mode)
closeBtn?.addEventListener("click", () => {
  document.body.classList.remove("show-chatbot");
});

// Handle preprocess button click
preprocessBtn?.addEventListener("click", preprocessDocuments);
