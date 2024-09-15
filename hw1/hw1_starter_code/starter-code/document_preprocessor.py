"""
This is the template for implementing the tokenizer for your search engine.
You will be testing some tokenization techniques and build your own tokenizer.
"""
import nltk
import spacy
from nltk.tokenize import RegexpTokenizer


class Tokenizer:
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        A generic class for objects that turn strings into sequences of tokens.
        A tokenizer can support different preprocessing options or use different methods
        for determining word breaks.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
        """
        # TODO: Save arguments that are needed as fields of this class
        self.lowercase = lowercase 
        self.multiword_expressions = multiword_expressions
    
    def postprocess(self, input_tokens: list[str]) -> list[str]:
        """
        Performs any set of optional operations to modify the tokenized list of words such as
        lower-casing and multi-word-expression handling. After that, return the modified list of tokens.

        Args:
            input_tokens: A list of tokens

        Returns:
            A list of tokens processed by lower-casing depending on the given condition
        """
        # TODO: Add support for lower-casing and multi-word expressions     
        if input_tokens == []:
            return []
           
        if self.multiword_expressions:
            # I want to sort the array by the word length from longest to shortest, thus "Taylor Swift" Tour will be detect ahead of "Taylor Swift"
            sorted_multiword_expressions = sorted(self.multiword_expressions, key=len, reverse=True)
            text = ' '.join(input_tokens)
            for multi_exp in sorted_multiword_expressions:
                if multi_exp in text:
                    text = text.replace(multi_exp, multi_exp.replace(" ", "$"))
            output_tokens = text.split(' ')
            output_tokens = [token.replace("$", " ") for token in output_tokens]
            if self.lowercase:
                output_tokens = [token.lower() for token in output_tokens]
            return output_tokens
        
        elif self.lowercase:
            input_tokens = [token.lower() for token in input_tokens]
        
        return input_tokens
    
    def tokenize(self, text: str) -> list[str]:
        """
        Splits a string into a list of tokens and performs all required postprocessing steps.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        # You should implement this in a subclass, not here
        raise NotImplementedError('tokenize() is not implemented in the base class; please use a subclass')


class SplitTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses the split function to tokenize a given string.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)

    def tokenize(self, text: str) -> list[str]:
        """
        Split a string into a list of tokens using whitespace as a delimiter.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        tokens = text.split()
        return self.postprocess(tokens)


class RegexTokenizer(Tokenizer):
    def __init__(self, token_regex: str = '\w+', lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses NLTK's RegexpTokenizer to tokenize a given string.

        Args:
            token_regex: Use the following default regular expression pattern: '\w+'
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)
        # TODO: Save a new argument that is needed as a field of this class
        self.token_regex = token_regex  
        
        # TODO: Initialize the NLTK's RegexpTokenizer 
        self.regex_tokenizer = RegexpTokenizer(token_regex)

    def tokenize(self, text: str) -> list[str]:
        """
        Uses NLTK's RegexTokenizer and a regular expression pattern to tokenize a string.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        # TODO: Tokenize the given text and perform postprocessing on the list of tokens
        #       using the postprocess function
        tokens = self.regex_tokenizer.tokenize(text)
        return self.postprocess(tokens)


class SpaCyTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Use a spaCy tokenizer to convert named entities into single words. 
        Check the spaCy documentation to learn about the feature that supports named entity recognition.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)
        self.spacy_tokenizer = spacy.load('en_core_web_sm')
        self.spacy_tokenizer.add_pipe("merge_entities")
        

    def tokenize(self, text: str) -> list[str]:
        """
        Use a spaCy tokenizer to convert named entities into single words.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        document = self.spacy_tokenizer(text)
        tokens = [token.text for token in document]
        return self.postprocess(tokens) # 这里可能含有标点符号
        


# Don't forget that you can have a main function here to test anything in the file
if __name__ == '__main__':
    pass
