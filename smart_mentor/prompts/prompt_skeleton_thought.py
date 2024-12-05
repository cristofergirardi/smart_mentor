
class PromptSkeleton():

    def __init__(self, question):
        self.question = question
    
    @property
    def first_think(self):
        return '''
            Regarding the user question:
            - Extract the main question;
            - There are some tips called "Dicas";
            - There are some input data, on that user need to add in this question;
            - Which kind of question it is;
            - What output is expected;
            - What goal is expected.
        '''
    
    @property
    def second_think(self):
        return f'''
           Generate a code python that performs the follwing task:
           {self.question}.
           Using the information that you received to answer the question asked by user. 
           Include comments in the code for clarity.
        '''
    
    @property
    def third_think(self):
        return f'''
           If there are some samples, it is to evaluate those samples and create your code.
        '''