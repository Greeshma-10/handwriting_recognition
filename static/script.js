const form = document.getElementById('ocr-form');
const resultDiv = document.getElementById('result');

// Function to display an error pop-up with a custom error message
function showErrorPopup(message) {
    const popup = document.createElement('div');
    popup.style.position = 'fixed';
    popup.style.top = '50%';
    popup.style.left = '50%';
    popup.style.transform = 'translate(-50%, -50%)';
    popup.style.width = '300px';
    popup.style.padding = '20px';
    popup.style.backgroundColor = '#ffffff';
    popup.style.boxShadow = '0 10px 30px rgba(0, 0, 0, 0.3)';
    popup.style.borderRadius = '12px';
    popup.style.textAlign = 'center';
    popup.style.zIndex = '9999';

    const cross = document.createElement('div');
    cross.style.width = '100px';
    cross.style.height = '100px';
    cross.style.margin = '0 auto 20px';
    cross.style.backgroundImage = 'url(https://img.icons8.com/color/100/error--v1.png)';
    cross.style.backgroundSize = 'contain';
    cross.style.backgroundRepeat = 'no-repeat';
    cross.style.backgroundPosition = 'center';
    popup.appendChild(cross);

    const messageElement = document.createElement('p');
    messageElement.textContent = message;
    messageElement.style.fontSize = '1.2rem';
    messageElement.style.fontWeight = 'bold';
    messageElement.style.color = '#e63946';
    popup.appendChild(messageElement);

    const closeButton = document.createElement('button');
    closeButton.textContent = 'Close';
    closeButton.style.marginTop = '20px';
    closeButton.style.padding = '10px 20px';
    closeButton.style.backgroundColor = '#e63946';
    closeButton.style.color = '#fff';
    closeButton.style.border = 'none';
    closeButton.style.borderRadius = '8px';
    closeButton.style.cursor = 'pointer';
    closeButton.style.fontSize = '1rem';
    closeButton.style.boxShadow = '0 5px 15px rgba(230, 57, 70, 0.4)';
    closeButton.addEventListener('click', () => {
        document.body.removeChild(popup);
    });
    popup.appendChild(closeButton);

    document.body.appendChild(popup);
}

// Form submission handler
form.addEventListener('submit', async (event) => {
    event.preventDefault();

    const formData = new FormData(form);
    const response = await fetch('/process-image', {
        method: 'POST',
        body: formData,
    });

    if (response.ok) {
        const data = await response.json();
        resultDiv.innerHTML = data.text || 'No text detected.';
    } else {
        const errorData = await response.json();
        // Show the error pop-up if processing fails
        showErrorPopup(errorData.error || 'Error processing image!');
    }
});
