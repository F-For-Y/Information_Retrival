import unittest
from unittest.mock import MagicMock
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from document_preprocessor import RegexTokenizer
import document_preprocessor


############ =======Test RegexTokenizer=========== ############
class TestRegexTokenizer(unittest.TestCase):
    """Test RegexTokenizer."""
    def test_split_empty_doc(self):
        """Test tokenizing an empty document."""
        text = ""
        expected_tokens = []
        tokenizer = RegexTokenizer('\\w+')
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)

    def test_one_token_doc(self):
        """Test tokenizing an one-token document."""
        text = "Michigan"
        expected_tokens = ["michigan"]
        tokenizer = RegexTokenizer('\\w+')
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)

    def test_doc_with_punc(self):
        """Test tokenizing a document with punctuations."""
        text = "RegexTokenizer can split on punctuation, like this: test!"
        expected_tokens = ['regextokenizer', 'can', 'split', 'on', 'punctuation', 'like', 'this', 'test']
        tokenizer = RegexTokenizer('\\w+')
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)

    def test_unicode_text(self):
        """Test tokenizing a text with an emoji."""
        text = "Welcome to the United States 🫥"
        expected_tokens = ['welcome', 'to', 'the', 'united', 'states']
        tokenizer = RegexTokenizer('\\w+')
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)
        
############ =======Test Doc2QueryAugmenter=========== ############
class TestDoc2QueryAugmenter(unittest.TestCase):
    def setUp(self):
        self.text = '''As a 21-year-old master\'s degree student at the Massachusetts Institute of Technology (MIT),
        he wrote his thesis demonstrating that electrical applications of Boolean algebra could construct any 
        logical numerical relationship.'''
        self.query = 'Who came up with the logic behind logic gates?'
        self.mockTokenizer = MagicMock(spec=['encode', 'decode'])
        self.mockTokenizer.decode.return_value = self.query
        self.mockModel = MagicMock(spec= ['generate','to'])
        self.mockModel.to.return_value = self.mockModel
        
        document_preprocessor.T5ForConditionalGeneration = MagicMock(spec=['from_pretrained'])
        document_preprocessor.T5ForConditionalGeneration.from_pretrained.return_value = self.mockModel
        
        document_preprocessor.T5Tokenizer = MagicMock(spec=['from_pretrained'])
        document_preprocessor.T5Tokenizer.from_pretrained.return_value = self.mockTokenizer


    def test_query_generation_with_no_prefix(self) -> None:
        d2q = document_preprocessor.Doc2QueryAugmenter('eecs-549/claude-shannon')
        query_count = 10
        self.mockModel.generate.return_value = [0 for _ in range(query_count)]
        queries = d2q.get_queries(self.text, query_count)
        self.assertEqual(len(queries), query_count, msg=f'Expected {query_count} queries as output but received {len(queries)}')
        for q in queries:
            self.assertEqual(q, self.query, f'Expected query to be "{self.query}" but received "{q}"')
        document_preprocessor.T5ForConditionalGeneration.from_pretrained.assert_called_once_with('eecs-549/claude-shannon')
        document_preprocessor.T5Tokenizer.from_pretrained.assert_called_once_with('eecs-549/claude-shannon')
        args = self.mockTokenizer.encode.call_args.args
        self.assertEqual(self.text, args[0], 'Expected input document different from the document processed.')
        self.mockModel.generate.assert_called_once()
        self.mockTokenizer.decode.assert_called()
    
    def test_query_generation_with_prefix(self) -> None:
        d2q = document_preprocessor.Doc2QueryAugmenter('eecs-549/claude-shannon')
        query_count = 10
        prefix = 'Some random prompt '
        self.mockModel.generate.return_value = [0 for _ in range(query_count)]
        queries = d2q.get_queries(self.text, query_count, prefix)
        self.assertEqual(len(queries), query_count, msg=f'Expected {query_count} queries as output but received {len(queries)}')
        for q in queries:
            self.assertEqual(q, self.query, f'Expected query to be "{self.query}" but received "{q}"')
        document_preprocessor.T5ForConditionalGeneration.from_pretrained.assert_called_once_with('eecs-549/claude-shannon')
        document_preprocessor.T5Tokenizer.from_pretrained.assert_called_once_with('eecs-549/claude-shannon')
        args = self.mockTokenizer.encode.call_args.args
        self.assertEqual(f'{prefix}{self.text}', args[0], 'Expected input document different from the document processed.')
        self.mockModel.generate.assert_called_once()
        self.mockTokenizer.decode.assert_called()

if __name__ == '__main__':
    unittest.main()

