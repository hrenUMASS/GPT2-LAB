def while_replace(string):
    while '  ' in string:
        string = string.replace('  ', ' ')
    return string


def data_mapping(fp1, fp2, save_path, read_lines=10000000):
    # conn = sqlite3.connect(db)
    # c = conn.cursor()
    with open(fp1, 'r') as f1, open(fp2, 'r') as f2, open(save_path, 'w+') as save, open(save_path + '.temp',
                                                                                         'w+') as save_temp:
        buff = build_database(f1, read_lines=read_lines)
        indexes = set()
        max_ind = 0
        origin = False
        while len(buff) != 0:
            save.seek(0)
            save_temp.seek(0)
            index = 0
            # print("ori {} f2 {}".format(ori.name, f2.name))
            if origin:
                f2 = save
                save, save_temp = save_temp, save
            for line in f2:
                if not origin:
                    max_ind += 1
                # if index not in indexes:
                line_id = line.replace(' ', '')
                line = buff.get(line_id, line)
                indexes.add(index)
                index += 1
                line += '' if line[-1] == '\n' else '\n'
                save.write(line)
            origin = True
            save.truncate()
            buff = build_database(f1, read_lines=read_lines)
        print('save: ', save.name, 'index: ', len(indexes) / max_ind)
        f2.seek(0)
        save.seek(0)
        line = f2.readline()
        line2 = save.readline()
        same = 0
        print('-------------------------------------')
        while line != '' or line2 != '':
            if line.replace(' ', '') != line2.replace(' ', ''):
                print(line, line2)
            if line != '':
                line = f2.readline()
            if line2 != '':
                line2 = save.readline()
            same += 1
        print(same / max_ind)
        print('save: ', save.name, 'index: ', len(indexes) / max_ind)
        # print('ori: {}, use: {}'.format(ori.name, use.name))

    # conn.close()


def build_database(f, read_lines=10000000):
    # conn = sqlite3.connect(fp1 + '.db')
    # c = conn.cursor()
    # c.execute("""
    #     CREATE TABLE IF NOT EXISTS t1 (
    #         id varchar,
    #         sent varchar
    #     );""")
    line = f.readline()
    index = 0
    cur, tot = f.tell(), f.seek(0, 2)
    f.seek(cur)
    buffer = {}
    while line:
        index += 1
        line = while_replace(line.split('\t')[0]) + '\n'
        line_id = line.replace(' ', '')
        buffer[line_id] = line

        if index == read_lines:
            print(f.tell() / tot)
            break
            # c.execute('BEGIN TRANSACTION')
            # for item in buffer:
            # c.execute("INSERT OR IGNORE INTO t1 (id, sent) VALUES (?, ?)", item)
            # c.execute('COMMIT')
        line = f.readline()
    return buffer
    # c.execute("CREATE UNIQUE INDEX t1i on t1 (id)")
    # conn.commit()
    # conn.close()


if __name__ == '__main__':
    data_path = '/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/'
    f1 = data_path + 'wiki2016_both.txt'
    f2 = '/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/wiki2016_nchunk_entity_agg/wiki2016_sents'
    save = '/iesl/canvas/hren/gpt2_wiki_lab/data/wiki2016_sents_mapped'
    # data_path = '../../data/wiki2016_'
    # f1 = data_path + 'both.txt'
    # f2 = data_path + '_sents'
    # save = data_path + '_sents_mapped'
    data_mapping(f1, f2, save)
