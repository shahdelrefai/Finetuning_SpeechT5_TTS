function createScrollable(elementId, range, pad = 2) {
    const element = document.getElementById(elementId);
    let value = 0;

    const updateDisplay = () => {
      const val = value.toString()
      element.querySelector('span').textContent = val;
    };

    const increment = () => {
      value = (value + 1) % range;
      updateDisplay();
    };

    const decrement = () => {
      value = (value - 1 + range) % range;
      updateDisplay();
    };

    // Mouse scroll event
    element.addEventListener('wheel', (event) => {
      event.preventDefault();
      event.deltaY > 0 ? increment() : decrement();
    });

    // Focus and keyboard navigation
    element.addEventListener('focus', () => {
      element.classList.add('selected');
    });

    element.addEventListener('blur', () => {
      element.classList.remove('selected');
    });

    element.addEventListener('keydown', (event) => {
      if (event.key === 'ArrowUp') {
        event.preventDefault();
        increment();
      } else if (event.key === 'ArrowDown') {
        event.preventDefault();
        decrement();
      }
    });

    updateDisplay();
  }



createScrollable('speakers', 20);
