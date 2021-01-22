import os
import codecs
import configparser


class Config:

    def __init__(self, config_file_path="./src/model/conf_bert_gnn_lstm.ini"):
        self.config = configparser.ConfigParser()
        self.config.read(config_file_path)
        self.sections = self.config.sections()

        self.config_dicts = dict()

        for section_name in self.sections:
            section_dict = dict(self.config.items(section_name))
            for (key, value) in section_dict.items():
                # 转化为整型数字
                try:
                    section_dict[key] = int(value)
                except ValueError:
                    pass

                # 转化为布尔型
                if value.lower() in ['true', 'false']:
                    if value.lower() == 'true':
                        section_dict[key] = True
                    else:
                        section_dict[key] = False

            self.config_dicts[section_name] = section_dict

        # self.print_config()

    def reset_config(self, args):

        if args.data_path != '':
            self.config_dicts['preprocess']['data_path'] = args.data_path
        if args.save_model_name != '':
            self.config_dicts['model']['save_model_name'] = args.save_model_name
        if args.train_batch_size:
            self.config_dicts['model']['train_batch_size'] = args.train_batch_size
        if args.epochs:
            self.config_dicts['model']['epochs'] = args.epochs
        if args.num_mid_layers is not None:
            self.config_dicts['model']['num_mid_layers'] = args.num_mid_layers

    def print_config(self, output_file=None):
        dump_string = "-" * 15 + " dump begin " + "-" * 15 + "\n"

        for section_name in self.sections:
            dump_string += "%s: \n" % (section_name)
            for (key, value) in self.config_dicts[section_name].items():
                dump_string += "  %s : %s\n" % (key, str(value))
            dump_string += "\n"

        dump_string += "-" * 15 + " dump finished " + "-" * 15 + "\n\n"

        if output_file is None:
            print(dump_string)
        else:
            with codecs.open(output_file, "w", encoding="utf-8") as writer:
                writer.write(dump_string)

if __name__ == "__main__":
    config = Config()
    config.print_config()

