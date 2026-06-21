#IMPORTANT!!!! IN ORDER TO USE PLEASE INSTALL pycryptodome by using:
#pip install pycryptodome

from Crypto.PublicKey import RSA
#from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5

class RSA_CLASS:
    def __init__(self):
        """
        פעולה בונה. יוצרת את המשתנים הבסיסייים לRSA דרך RSA.GENERATE
        יוצר מפתח באורך 2048 ביטים ושם את המפתחות במשתנים משלהם
        יוצר נוסחאת הצפנה דרך PRIVATE_RSA וCIPHER_RSA_DECRYPT כאשר CIPHER_RSA_DECRYPT הוא אובייקט הפענוח שלנו
        """
        self.key = RSA.generate(2048) #יצירת אובייקט RSAKEY יש בתכונות שלו את המפתח הציבורי\מפתח פרטי, P, Q, N וכו
        self.private_key = self.key.export_key() #יצירת תכונה למפתח הפרטי לנוחות
        self.public_key = self.key.public_key().export_key() #יצירת תכונה למפתח הפרטי לנוחות

        self.other_public = None
        self.cipher_rsa_encrypt = None #אובייקט הצפנה. מאפשר לנו להצפין בהצפנה מסוימת עם מפתח מוגדר

        self.private_rsa = RSA.import_key(self.private_key) # הוספת המפתח הפרטי לאובייקט RSA. אי אפשר לעבוד עם בתים אז צריך להמיר

        self.cipher_rsa_decrypt = PKCS1_v1_5.new(self.private_rsa) #יצירת אובייקט הפענוח של RSA לפי תקן PKCS1_V1_5 בשביל הריפוד

    def set_other_public(self, other_public_key):
        """
        פעולת SET שמעדכנת את התכונה ששומרת על המפתח הציבורי של הצד המקבל
        היא גם קוראת לפעולה שמעדכנת את אובייקט ההצפנה לצד המקבל בCIPHER_RSA_ENCRYPT
        :param other_public_key: המפתח הציבורי של הצד המקבל בתקשרות
        """
        if isinstance(other_public_key, str):
            other_public_key = other_public_key.encode('utf-8')
        self.other_public = RSA.import_key(other_public_key) #הוספת המפתח הציבורי של הצד המקבל לאובייקט RSA. אי אפשר לעבוד עם בתים אז צריך להמיר

        self.set_encryption_object() #קריאה


    def set_encryption_object(self):
        self.cipher_rsa_encrypt = PKCS1_v1_5.new(self.other_public) #הגדרת אובייקט ההצפנה של RSA לפי תקן PKCS1_V1_5 בשביל הריפוד עבור המפתח הציבורי של הצד המקבל

    def encrypt_RSA(self,message):
        """
        מצפינה טקסט דרך אובייקט ההצפנה שיצרנו עבור הצד המקבל
        :param message:ההודעה שרוצים לשלוח
        :return: הודעה מוצפנת בבתים מוכנה לשליחה
        """
        #בדיקת המרה לבתים
        if not isinstance(message, bytes):
            if isinstance(message, str):
                message = message.encode()
            else:
                message = bytes(message)

        return self.cipher_rsa_encrypt.encrypt(message) #שימוש באובייקט ההצפנה שיצרנו על מנת להצפין עם המפתח הציבורי של הצד המקבל

    def decrypt_RSA(self,encrypted_message):
        """
        מפענחת טקסט-צופן דרך אובייקט הפענוח שיצרנו עם המפתח הפרטי שלנו
        :param encrypted_message: מסר מוצפן
        :return: מסר טקסט נקי וקריא
        """
        return self.cipher_rsa_decrypt.decrypt(encrypted_message,None) #שימוש באובייקט הפענוח שיצרנו על מנת לפענח עם המפתח הפרטי שלנו