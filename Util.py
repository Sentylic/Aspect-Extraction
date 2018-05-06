import json
import xmltodict

class Util:
    @staticmethod
    def convertXMLToJson(xml_file_name, json_file_name):

        with open(xml_file_name, 'r') as f:
            xmlString = f.read()

        print("XML input (sample.xml):")
        print(xmlString)

        jsonString = json.dumps(xmltodict.parse(xmlString), indent=4)

        print("\nJSON output(output.json):")
        print(jsonString)

        with open(json_file_name, 'w') as f:
            f.write(jsonString)
