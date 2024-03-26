from numpy.lib.function_base import append
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import VectorParams, Distance
import random
import socket
import threading
import os

# Initialize OpenAI and Qdrant clients with API keys
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

def retrieve_recommended_context(prompt):
    # Perform a search in Qdrant using embeddings from OpenAI
    responses = qdrant_client.search(
        collection_name=collection_name,
        query_vector=openai_client.embeddings.create(
            input=prompt,
            model="text-embedding-ada-002",
        ).data[0].embedding,
        with_vectors=False,
        with_payload=True,
        limit=3
    )

    # Extract texts from the search responses
    texts = [response.payload["text"] for response in responses]
    return "\n".join(texts)

def generate_recommendation(conversations, main_topics):
    # Randomly choose a topic from the provided list
    topic = random.choice(main_topics)
    # Add system and user messages to the conversation
    conversations.append({"role": "system", "content": "Your role is to generate a quiz for a cloud computing module."})
    conversations.append({"role": "user", "content": f"Recommend and describe one subtopic that is related to \"{topic}\""})
    # Generate a completion using OpenAI's chat model
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversations,
        temperature=1.5
    )

    # Add the assistant's response to the conversation
    conversations.append({"role": "assistant", "content": response.choices[0].message.content})
    return response.choices[0].message.content

def generate_quiz(context, conversations):
    # Copy conversations and add a new user message
    tmp_conversations = conversations.copy()
    tmp_conversations.append({"role": "user", "content": f"{context}\n\nBased on the context provided above, generate only 1 multiple choice question or 1 short answer question. Do not provide the answer. Quiz takers are not aware of the context provided. Never ask the same question"})
    # Generate a completion using OpenAI's chat model
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=tmp_conversations,
        temperature=1.2
    )

    # Add the assistant's generated question to the conversation
    conversations.append({"role": "assistant", "content": response.choices[0].message.content})
    return response.choices[0].message.content

def verify_answer(answer, conversations):
    # Add user's answer to the conversation and generate a completion
    conversations.append({"role": "user", "content": f"My answer is: {answer}\n\nIf wrong, provide the correct answer and further elaborate on the concept."})
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=conversations,
        temperature=0.2
    )

    # Add the assistant's feedback to the conversation
    conversations.append({"role": "assistant", "content": response.choices[0].message.content})
    return response.choices[0].message.content

def socket_listen(host, port):
    # Set up a listening socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"Server listening on {host}:{port}")
        try:
            # Accept connections and handle them in separate threads
            while True:
                conn, addr = s.accept()
                print(f"Connection from {addr}")
                client_handler = threading.Thread(
                    target=handle_client_connection,
                    args=(conn, addr,)
                )
                client_handler.start()
        except KeyboardInterrupt:
            # Gracefully shutdown the server on keyboard interrupt
            s.close()
            print("Server shutting down gracefully...")

def handle_client_connection(conn, addr):
    # Handle individual client connection
    individual_conversations = []
    try:
        # Send welcome message to the client
        conn.sendall("Welcome to".encode())
        conn.sendall("""\n
            _______  _______   ___    _______  _______  _______
           (  ____ \(  ____ \ /   )  (  __   )(  ____ \/ ___   )
           | (    \/| (    \// /) |  | (  )  || (    \/\/   )  |
           | (_____ | |     / (_) (_ | | /   || (____      /   )
           (_____  )| |    (____   _)| (/ /) |(_____ \   _/   /
                 ) || |         ) (  |   / | |      ) ) /   _/
           /\____) || (____/\   | |  |  (__) |/\____) )(   (__/\\
           \_______)(_______/   (_)  (_______)\______/ \_______/

            _______  _______  _______    _______          _________ _______
           (  ____ )(  ___  )(  ____ )  (  ___  )|\     /|\__   __// ___   )
           | (    )|| (   ) || (    )|  | (   ) || )   ( |   ) (   \/   )  |
           | (____)|| |   | || (____)|  | |   | || |   | |   | |       /   )
           |  _____)| |   | ||  _____)  | |   | || |   | |   | |      /   /
           | (      | |   | || (        | | /\| || |   | |   | |     /   /
           | )      | (___) || )        | (_\ \ || (___) |___) (___ /   (_/\\
           |/       (_______)|/         (____\/_)(_______)\_______/(_______/

        \n""".encode())
        # Process data received from the client
        process_received_data(conn, individual_conversations)
    except ConnectionResetError:
        print("Connection aborted by client.")
    finally:
        # Close the connection after handling it
        conn.close()
        print(f"Connection with {addr} closed.")

def process_received_data(conn, conversations):
    # Process data received from the client and interact using the OpenAI model
    for _ in range(5):
        conn.sendall("Generating question...\n\n".encode())
        recommendation = generate_recommendation(conversations, main_topics)

        context = retrieve_recommended_context(recommendation)

        question = generate_quiz(context, conversations)
        conn.sendall(f"Question: {question}\n".encode())

        conn.sendall("\nPlease provide your answer: ".encode())
        answer_input = conn.recv(1024).decode().strip()
        conn.sendall("Verifying answer...\n\n".encode())
        feedback = verify_answer(answer_input, conversations)
        conn.sendall(f"{feedback}\n\n".encode())

        conn.sendall("Display referenced data? (y/n): ".encode())
        is_display = conn.recv(1024).decode().strip()
        if is_display.lower() == 'y':
            conn.sendall(f"\nReferences:\n{context}\n".encode())

        conn.sendall(("\n" + "="*20 + "\n\n").encode())

    conn.sendall("The End :)".encode())

# Collection name for Qdrant search
collection_name = "sc4052-lecture"

# List of main topics for generating recommendations
main_topics = [
    'Basics, IaaS, PaaS, SaaS', 'Data Center Networking–Basics, Topology',
    'Virtualization in Cloud', 'Cloud CPU Scheduling', 'Crowdsourcing in Cloud',
    'Cloud Security', 'CAP Theorem', 'PageRank Algorithm'
]

# Start the server listening on localhost port 4052
socket_listen("0.0.0.0", 4052)
