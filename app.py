from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from brain import add_documents, query_llm
from processor import process_pdf


ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class RAGApp(ctk.CTk):
	def __init__(self) -> None:
		super().__init__()
		self.title("Local RAG Chatbot")
		self.geometry("980x720")
		self.minsize(860, 620)

		self._build_layout()

	def _build_layout(self) -> None:
		self.grid_columnconfigure(0, weight=1)
		self.grid_rowconfigure(1, weight=1)

		header_frame = ctk.CTkFrame(self, corner_radius=16)
		header_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
		header_frame.grid_columnconfigure(0, weight=1)

		title_label = ctk.CTkLabel(
			header_frame,
			text="Local Private RAG Chatbot",
			font=ctk.CTkFont(size=24, weight="bold"),
		)
		title_label.grid(row=0, column=0, padx=18, pady=(16, 4), sticky="w")

		subtitle_label = ctk.CTkLabel(
			header_frame,
			text="All processing stays on your machine with Ollama and ChromaDB.",
			text_color="gray70",
		)
		subtitle_label.grid(row=1, column=0, padx=18, pady=(0, 16), sticky="w")

		body_frame = ctk.CTkFrame(self, corner_radius=16)
		body_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
		body_frame.grid_columnconfigure(0, weight=1)
		body_frame.grid_rowconfigure(0, weight=1)

		self.chat_box = ctk.CTkTextbox(body_frame, wrap="word")
		self.chat_box.grid(row=0, column=0, padx=16, pady=16, sticky="nsew")
		self.chat_box.configure(state="disabled")

		input_frame = ctk.CTkFrame(self, corner_radius=16)
		input_frame.grid(row=2, column=0, padx=20, pady=(10, 20), sticky="ew")
		input_frame.grid_columnconfigure(0, weight=1)

		self.message_entry = ctk.CTkEntry(
			input_frame,
			placeholder_text="Ask a question about your uploaded PDFs...",
		)
		self.message_entry.grid(row=0, column=0, padx=(16, 10), pady=16, sticky="ew")
		self.message_entry.bind("<Return>", lambda event: self.send_message())

		send_button = ctk.CTkButton(input_frame, text="Send", command=self.send_message)
		send_button.grid(row=0, column=1, padx=(0, 10), pady=16)

		upload_button = ctk.CTkButton(
			input_frame,
			text="Upload PDF",
			command=self.upload_pdf,
		)
		upload_button.grid(row=0, column=2, padx=(0, 16), pady=16)

		self._append_message("assistant", "Upload a PDF to start building the local knowledge base.")

	def _append_message(self, role: str, message: str) -> None:
		self.chat_box.configure(state="normal")
		self.chat_box.insert("end", f"{role.title()}: {message}\n\n")
		self.chat_box.see("end")
		self.chat_box.configure(state="disabled")

	def upload_pdf(self) -> None:
		file_path = filedialog.askopenfilename(
			title="Select a PDF",
			initialdir=str(Path("data")),
			filetypes=[("PDF files", "*.pdf")],
		)
		if not file_path:
			return

		try:
			documents = process_pdf(file_path)
			chunk_count = add_documents(documents)
			self._append_message(
				"assistant",
				f"Indexed {chunk_count} text chunks from {Path(file_path).name}.",
			)
			messagebox.showinfo("Upload complete", f"Processed {chunk_count} chunks.")
		except Exception as error:
			messagebox.showerror("Upload failed", str(error))

	def send_message(self) -> None:
		user_input = self.message_entry.get().strip()
		if not user_input:
			return

		self.message_entry.delete(0, "end")
		self._append_message("user", user_input)

		try:
			answer = query_llm(user_input)
		except Exception as error:
			answer = f"Error: {error}"

		self._append_message("assistant", answer)


def main() -> None:
	app = RAGApp()
	app.mainloop()


if __name__ == "__main__":
	main()
