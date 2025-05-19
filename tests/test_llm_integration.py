#!/usr/bin/env python
# Cosmic Market Oracle - LLM Integration Tests

"""
Tests for the LLM Knowledge Integration System.

This module contains unit tests for the various components of the
LLM Knowledge Integration System.
"""

import os
import sys
import unittest
import json
from datetime import datetime
from unittest.mock import MagicMock, patch
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm_integration.knowledge_base import AstrologicalKnowledgeBase, Document
from src.llm_integration.prompt_engineering import PromptTemplate, PromptLibrary
from src.llm_integration.retrieval_augmentation import LLMProvider, RAGSystem
from src.llm_integration.reasoning_chains import ReasoningStep, ReasoningChain
from src.llm_integration.output_formats import OutputFormatter
from src.llm_integration.explanation_module import ExplanationGenerator
from src.llm_integration.conversation import Message, Conversation, ConversationalInterface
from src.llm_integration.report_generation import ReportGenerator
from src.llm_integration.continuous_learning import ContinuousLearningPipeline


class TestDocument(unittest.TestCase):
    """Tests for the Document class."""
    
    def test_document_creation(self):
        """Test document creation."""
        content = "Test content"
        metadata = {"author": "Test Author", "source": "Test Source"}
        
        doc = Document(content=content, metadata=metadata)
        
        self.assertEqual(doc.content, content)
        self.assertEqual(doc.metadata, metadata)
        self.assertIsNotNone(doc.doc_id)
    
    def test_document_chunking(self):
        """Test document chunking."""
        content = "A" * 2000  # 2000 characters
        metadata = {"author": "Test Author"}
        
        doc = Document(content=content, metadata=metadata)
        chunks = doc.chunk(chunk_size=1000, overlap=200)
        
        self.assertEqual(len(chunks), 3)  # 2 full chunks + 1 small chunk
        self.assertEqual(len(chunks[0]["text"]), 1000)
        self.assertEqual(len(chunks[1]["text"]), 1000)
        self.assertEqual(chunks[0]["text"][-200:], chunks[1]["text"][:200])  # Check overlap
    
    def test_document_serialization(self):
        """Test document serialization."""
        content = "Test content"
        metadata = {"author": "Test Author"}
        
        doc = Document(content=content, metadata=metadata)
        doc_dict = doc.to_dict()
        
        self.assertEqual(doc_dict["content"], content)
        self.assertEqual(doc_dict["metadata"], metadata)
        
        # Test deserialization
        new_doc = Document.from_dict(doc_dict)
        self.assertEqual(new_doc.content, doc.content)
        self.assertEqual(new_doc.metadata, doc.metadata)
        self.assertEqual(new_doc.doc_id, doc.doc_id)


class TestAstrologicalKnowledgeBase(unittest.TestCase):
    """Tests for the AstrologicalKnowledgeBase class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.kb_path = os.path.join(self.temp_dir, "kb")
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    @patch('src.llm_integration.knowledge_base.SentenceTransformer')
    @patch('src.llm_integration.knowledge_base.faiss')
    def test_knowledge_base_creation(self, mock_faiss, mock_transformer):
        """Test knowledge base creation."""
        # Mock the embedding model
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model
        
        # Mock FAISS index
        mock_index = MagicMock()
        mock_faiss.IndexFlatL2.return_value = mock_index
        
        # Create knowledge base
        kb = AstrologicalKnowledgeBase(base_path=self.kb_path)
        
        self.assertEqual(kb.base_path, Path(self.kb_path))
        self.assertEqual(kb.embedding_model, mock_model)
        self.assertEqual(kb.index, mock_index)
    
    @patch('src.llm_integration.knowledge_base.SentenceTransformer')
    @patch('src.llm_integration.knowledge_base.faiss')
    def test_add_document(self, mock_faiss, mock_transformer):
        """Test adding a document to the knowledge base."""
        # Mock the embedding model
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = [[0.1] * 384]
        mock_transformer.return_value = mock_model
        
        # Mock FAISS index
        mock_index = MagicMock()
        mock_faiss.IndexFlatL2.return_value = mock_index
        
        # Create knowledge base
        kb = AstrologicalKnowledgeBase(base_path=self.kb_path)
        
        # Add document
        content = "Test content"
        metadata = {"author": "Test Author"}
        
        doc_id = kb.add_document(content=content, metadata=metadata)
        
        self.assertIsNotNone(doc_id)
        self.assertIn(doc_id, kb.documents)
        mock_index.add.assert_called()
    
    @patch('src.llm_integration.knowledge_base.SentenceTransformer')
    @patch('src.llm_integration.knowledge_base.faiss')
    def test_search(self, mock_faiss, mock_transformer):
        """Test searching the knowledge base."""
        # Mock the embedding model
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = [[0.1] * 384]
        mock_transformer.return_value = mock_model
        
        # Mock FAISS index
        mock_index = MagicMock()
        mock_index.search.return_value = ([0.5], [[0]])
        mock_faiss.IndexFlatL2.return_value = mock_index
        
        # Create knowledge base
        kb = AstrologicalKnowledgeBase(base_path=self.kb_path)
        
        # Add document
        content = "Test content"
        metadata = {"author": "Test Author"}
        
        doc_id = kb.add_document(content=content, metadata=metadata)
        
        # Search
        results = kb.search(query="test", top_k=1)
        
        self.assertEqual(len(results), 1)
        mock_index.search.assert_called_with(mock_model.encode.return_value, 1)


class TestPromptEngineering(unittest.TestCase):
    """Tests for the prompt engineering module."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.prompts_path = os.path.join(self.temp_dir, "prompts")
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_prompt_template(self):
        """Test prompt template."""
        template = "Hello, {name}!"
        name = "Test Template"
        description = "Test description"
        
        prompt = PromptTemplate(template=template, name=name, description=description)
        
        self.assertEqual(prompt.template, template)
        self.assertEqual(prompt.name, name)
        self.assertEqual(prompt.description, description)
        
        # Test formatting
        formatted = prompt.format(name="World")
        self.assertEqual(formatted, "Hello, World!")
    
    def test_prompt_library(self):
        """Test prompt library."""
        library = PromptLibrary(base_path=self.prompts_path)
        
        # Create template
        template = PromptTemplate(
            template="Hello, {name}!",
            name="test_template",
            description="Test template"
        )
        
        # Add to library
        library.add_template(template)
        
        # Get template
        retrieved = library.get_template("test_template")
        
        self.assertEqual(retrieved.template, template.template)
        self.assertEqual(retrieved.name, template.name)
        
        # List templates
        templates = library.list_templates()
        self.assertEqual(len(templates), 1)
        self.assertEqual(templates[0]["name"], "test_template")


class TestRetrievalAugmentation(unittest.TestCase):
    """Tests for the retrieval augmentation module."""
    
    @patch('src.llm_integration.retrieval_augmentation.requests')
    def test_llm_provider(self, mock_requests):
        """Test LLM provider."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        
        mock_openai = MagicMock()
        mock_openai.ChatCompletion.create.return_value = mock_response
        
        with patch.dict('sys.modules', {'openai': mock_openai}):
            provider = LLMProvider(provider="openai", api_key="test_key")
            
            response = provider.generate("Test prompt")
            
            self.assertEqual(response, "Test response")
            mock_openai.ChatCompletion.create.assert_called()


class TestReasoningChains(unittest.TestCase):
    """Tests for the reasoning chains module."""
    
    def test_reasoning_step(self):
        """Test reasoning step."""
        # Create step
        step = ReasoningStep(
            name="test_step",
            description="Test step",
            prompt_template="Hello, {name}!",
            output_parser=lambda x: x.upper()
        )
        
        # Create mock LLM provider
        provider = MagicMock()
        provider.generate.return_value = "test response"
        
        # Execute step
        result = step.execute(provider, {"name": "World"})
        
        self.assertEqual(result["step_name"], "test_step")
        self.assertEqual(result["raw_output"], "test response")
        self.assertEqual(result["parsed_output"], "TEST RESPONSE")
        provider.generate.assert_called_with("Hello, World!")
    
    def test_reasoning_chain(self):
        """Test reasoning chain."""
        # Create steps
        step1 = ReasoningStep(
            name="step1",
            description="Step 1",
            prompt_template="Hello, {name}!",
            output_parser=lambda x: x.upper()
        )
        
        step2 = ReasoningStep(
            name="step2",
            description="Step 2",
            prompt_template="Response: {step1}",
            output_parser=lambda x: len(x)
        )
        
        # Create mock LLM provider
        provider = MagicMock()
        provider.generate.side_effect = ["test response", "second response"]
        
        # Create chain
        chain = ReasoningChain(
            name="test_chain",
            description="Test chain",
            steps=[step1, step2],
            llm_provider=provider
        )
        
        # Execute chain
        result = chain.execute({"name": "World"})
        
        self.assertEqual(result["chain_name"], "test_chain")
        self.assertEqual(len(result["steps"]), 2)
        self.assertEqual(result["step1"], "TEST RESPONSE")
        self.assertEqual(result["step2"], 15)  # Length of "second response"


class TestOutputFormats(unittest.TestCase):
    """Tests for the output formats module."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.temp_dir, "output")
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_output_formatter(self):
        """Test output formatter."""
        formatter = OutputFormatter(output_dir=self.output_path)
        
        # Test JSON formatting
        data = {"key": "value"}
        json_output = formatter.format_as_json(data)
        self.assertEqual(json.loads(json_output), data)
        
        # Test Markdown formatting
        template = "# {title}\n\n{content}"
        data = {"title": "Test", "content": "Test content"}
        md_output = formatter.format_as_markdown(data, template)
        self.assertEqual(md_output, "# Test\n\nTest content")
        
        # Test saving output
        path = formatter.save_output(data, "test_output", "json")
        self.assertTrue(os.path.exists(path))
        
        with open(path, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data, data)


class TestExplanationModule(unittest.TestCase):
    """Tests for the explanation module."""
    
    def test_explanation_generator(self):
        """Test explanation generator."""
        # Create mock LLM provider
        provider = MagicMock()
        provider.generate.return_value = "Test explanation"
        
        # Create explanation generator
        generator = ExplanationGenerator(llm_provider=provider)
        
        # Generate explanation
        prediction = {
            "direction": "up",
            "confidence": 0.8,
            "astrological_factors": [
                {"factor": "Jupiter-Saturn conjunction"}
            ]
        }
        
        explanation = generator.explain_prediction(prediction, level="beginner")
        
        self.assertEqual(explanation["explanation"], "Test explanation")
        self.assertEqual(explanation["level"], "beginner")
        provider.generate.assert_called()


class TestConversation(unittest.TestCase):
    """Tests for the conversation module."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.conversations_path = os.path.join(self.temp_dir, "conversations")
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_message(self):
        """Test message."""
        message = Message(role="user", content="Test message")
        
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "Test message")
        self.assertIsNotNone(message.message_id)
        
        # Test serialization
        message_dict = message.to_dict()
        
        self.assertEqual(message_dict["role"], "user")
        self.assertEqual(message_dict["content"], "Test message")
        
        # Test deserialization
        new_message = Message.from_dict(message_dict)
        
        self.assertEqual(new_message.role, message.role)
        self.assertEqual(new_message.content, message.content)
        self.assertEqual(new_message.message_id, message.message_id)
    
    def test_conversation(self):
        """Test conversation."""
        conversation = Conversation(title="Test Conversation")
        
        self.assertEqual(conversation.title, "Test Conversation")
        self.assertIsNotNone(conversation.conversation_id)
        
        # Add messages
        message1 = Message(role="user", content="Hello")
        message2 = Message(role="assistant", content="Hi there")
        
        conversation.add_message(message1)
        conversation.add_message(message2)
        
        self.assertEqual(len(conversation.messages), 2)
        
        # Get messages
        messages = conversation.get_messages()
        self.assertEqual(len(messages), 2)
        
        # Get message history
        history = conversation.get_message_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")
        
        # Test serialization
        conversation_dict = conversation.to_dict()
        
        self.assertEqual(conversation_dict["title"], "Test Conversation")
        self.assertEqual(len(conversation_dict["messages"]), 2)
        
        # Test deserialization
        new_conversation = Conversation.from_dict(conversation_dict)
        
        self.assertEqual(new_conversation.title, conversation.title)
        self.assertEqual(new_conversation.conversation_id, conversation.conversation_id)
        self.assertEqual(len(new_conversation.messages), 2)


class TestReportGeneration(unittest.TestCase):
    """Tests for the report generation module."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.reports_path = os.path.join(self.temp_dir, "reports")
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_report_generator(self):
        """Test report generator."""
        # Create mock LLM provider
        provider = MagicMock()
        provider.generate.return_value = json.dumps({
            "title": "Test Report",
            "content": "Test content"
        })
        
        # Create mock output formatter
        formatter = MagicMock()
        formatter.format_as_markdown.return_value = "# Test Report\n\nTest content"
        
        # Create report generator
        generator = ReportGenerator(
            llm_provider=provider,
            output_formatter=formatter,
            output_dir=self.reports_path
        )
        
        # Generate report
        market_data = {"symbol": "AAPL", "price": 150.0}
        planetary_data = {"jupiter": {"sign": "Aries"}}
        
        report = generator.generate_report(
            report_type="trader",
            market_data=market_data,
            planetary_data=planetary_data
        )
        
        self.assertEqual(report["report_type"], "trader")
        provider.generate.assert_called()
        formatter.format_as_markdown.assert_called()


class TestContinuousLearning(unittest.TestCase):
    """Tests for the continuous learning module."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.updates_path = os.path.join(self.temp_dir, "updates")
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    @patch('src.llm_integration.continuous_learning.AstrologicalKnowledgeBase')
    def test_continuous_learning_pipeline(self, mock_kb_class):
        """Test continuous learning pipeline."""
        # Create mock knowledge base
        mock_kb = MagicMock()
        mock_kb_class.return_value = mock_kb
        
        # Create data sources
        data_sources = {
            "test_source": {
                "type": "file",
                "path": os.path.join(self.temp_dir, "test_data.json")
            }
        }
        
        # Create test data
        test_data = [{"content": "Test content"}]
        
        with open(os.path.join(self.temp_dir, "test_data.json"), 'w') as f:
            json.dump(test_data, f)
        
        # Create pipeline
        pipeline = ContinuousLearningPipeline(
            knowledge_base=mock_kb,
            data_sources=data_sources,
            update_interval=24,
            output_dir=self.updates_path
        )
        
        # Test update
        with patch.object(pipeline, '_collect_from_file', return_value=test_data):
            results = pipeline.update()
            
            self.assertIn("test_source", results["sources"])
            mock_kb.add_document.assert_called()


if __name__ == '__main__':
    unittest.main()
