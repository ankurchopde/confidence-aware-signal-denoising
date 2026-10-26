# Contributing Guidelines

Thank you for your interest in contributing to the **Confidence-Aware AI for Sensor Signal Denoising** project!

We welcome contributions across signal processing algorithms, deep learning architectures, edge deployment optimizations, and documentation.

---

## 1. Development Workflow

1. **Fork the Repository:** Create a personal fork on GitHub.
2. **Clone and Branch:**
   ```bash
   git clone https://github.com/<your-username>/confidence-aware-signal-denoising.git
   cd confidence-aware-signal-denoising
   git checkout -b feature/your-feature-name
   ```
3. **Set Up Development Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```
4. **Implement Changes:** Write clean, modular Python code with type annotations and docstrings.
5. **Run the Test Suite:**
   ```bash
   pytest -v
   ```
6. **Submit a Pull Request:** Ensure your commit messages are descriptive and conform to standard open-source conventions.

---

## 2. Code Standards

- Follow PEP 8 style guidelines.
- Maintain comprehensive unit tests in `tests/` for any new signal generators, filters, or loss functions.
- Do not commit large dataset files or confidential conference materials into the Git tree.
