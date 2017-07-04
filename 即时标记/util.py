def lines(file):
    for line in file: yield line
    yield '\n'

def blocks(file):
    block = []
    for line in lines(file):
        if line.strip():  # not empty line 
            block.append(line)
        elif block: #empty line and not empty block
            yield ''.join(block).strip()
            block = []